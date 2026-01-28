# Machine Failure Prediction

This repository provides an end-to-end workflow for predicting machine failure while comparing various models and ultimately selecting XGBoost. 

## Project Overview

The project is structured to include a FastAPI service for online inference, a Docker container for deployment to AWS Lambda, and a local offline test script: validate_server.py. The process for this project include:
- Data preparation and cleaning
- Feature engineering and preprocessing
- Training and packaging the model (ML workflow)
- Serving predictions via a FastAPI service
- Containerizing the Lambda handler with Docker for deployment to AWS Lambda

The project includes a single serialized artifact that contains:
- DictVectorizer (dv)
- Trained XGBoost model (model)
- StandardScaler (sc)

All three objects are stored in machine_failure_prediction.pkl and are loaded by the API and Lambda handler during startup.


## Repository Structure

- Machine_Failure_Prediction.ipynb – Notebook for data exploration, feature processing, training, and artifact creation
- Machine_Failure_Prediction.py – Scripted version of parts of the notebook (if used)
- machine_failure_prediction.pkl – Packed artifact with dv, model, sc
- main.py – FastAPI app for online inference
- validate_server.py – Local client to test the FastAPI endpoint
- test_model.py – Local offline test of the artifact without FastAPI
- lambda_function.py – AWS Lambda handler for inference
- Dockerfile – Container definition for the Lambda deployment image (Python 3.12 base)
- pyproject.toml / uv.lock – Python dependencies managed with uv


## Data and Target

- Data: Machine_Downtime.csv (raw data used for training in the notebook)
- Task: Binary classification predicting machine failure


## Model Training Workflow (high level)

1. Data preparation
   - Load Machine_Downtime.csv
   - Clean and validate columns
   - Select features and handle missing values

2. Feature engineering and preprocessing
   - Numeric features are standardized using StandardScaler
   - DictVectorizer is used to convert feature dictionaries into a numeric feature matrix compatible with XGBoost

3. Model training
   - XGBoost binary classifier is trained on the transformed feature matrix
   - Hyperparameters can be tuned using cross-validation or grid/random search (see notebook)

4. Packaging artifacts
   - Persist DictVectorizer (dv), trained model, and StandardScaler (sc) in a single pickle file (machine_failure_prediction.pkl)

5. Inference pipeline (shared by API and Lambda)
   - Accept input features as JSON
   - Reconstruct a dict with the exact training-time feature names
   - Standardize numeric features using sc
   - Apply dv.transform to produce feature matrix
   - Run XGBoost prediction (0 or 1)


## Required Features for Inference

The service expects the following 12 features as inputs. For API consumers, simplified keys without units are accepted; internally they are mapped to the training-time feature names with units and standardized before prediction.

Simplified JSON keys (accepted by FastAPI):
- Hydraulic_Pressure
- Coolant_Pressure
- Air_System_Pressure
- Coolant_Temperature
- Hydraulic_Oil_Temperature
- Spindle_Bearing_Temperature
- Spindle_Vibration
- Tool_Vibration
- Spindle_Speed
- Voltage
- Torque
- Cutting

Training-time feature names (internal mapping):
- Hydraulic_Pressure(bar)
- Coolant_Pressure(bar)
- Air_System_Pressure(bar)
- Coolant_Temperature
- Hydraulic_Oil_Temperature(?C)
- Spindle_Bearing_Temperature(?C)
- Spindle_Vibration(?m)
- Tool_Vibration(?m)
- Spindle_Speed(RPM)
- Voltage(volts)
- Torque(Nm)
- Cutting(kN)


## FastAPI Inference Service

The main.py application exposes two endpoints:
- GET / – Health check
- POST /predict – Returns a JSON with prediction (0 or 1)

Run locally:
1. Ensure Python 3.10+ and dependencies from pyproject.toml are installed (via uv or pip). If using uv:
   - uv pip install --system --requirement pyproject.toml
2. Start the server:
   - uvicorn main:app --host 127.0.0.1 --port 8010 --reload
3. Test with the included client:
   - python validate_server.py

Example request payload (using simplified keys):
{
  "Hydraulic_Pressure": 71.04,
  "Coolant_Pressure": 6.93,
  "Air_System_Pressure": 6.28,
  "Coolant_Temperature": 25.6,
  "Hydraulic_Oil_Temperature": 46.0,
  "Spindle_Bearing_Temperature": 33.4,
  "Spindle_Vibration": 1.29,
  "Tool_Vibration": 26.49,
  "Spindle_Speed": 25892.0,
  "Voltage": 335.0,
  "Torque": 24.06,
  "Cutting": 3.58
}

Example curl:
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Hydraulic_Pressure": 71.04,
    "Coolant_Pressure": 6.93,
    "Air_System_Pressure": 6.28,
    "Coolant_Temperature": 25.6,
    "Hydraulic_Oil_Temperature": 46.0,
    "Spindle_Bearing_Temperature": 33.4,
    "Spindle_Vibration": 1.29,
    "Tool_Vibration": 26.49,
    "Spindle_Speed": 25892.0,
    "Voltage": 335.0,
    "Torque": 24.06,
    "Cutting": 3.58
  }'

Response:
{
  "prediction": 0,
  "count": 1
}

Notes
- The Pydantic model allows using simplified keys in the request. The service maps them to the exact training-time feature names internally before scaling and vectorization.
- The API loads the artifact once at startup for performance.


## Local Offline Test (no FastAPI)

Use test_model.py to sanity-check the artifact and pipeline without the API layer. It loads the pickle, standardizes inputs, vectorizes, and performs a single prediction.

Run:
- python test_model.py


## Packaging for AWS Lambda (Docker)

This repository uses the AWS Lambda Python 3.12 base image and uv for dependency installation. The image embeds the Lambda handler and the model artifact so it can be pushed to ECR and deployed as a container-based Lambda function.

Key files
- lambda_function.py – Lambda handler with the same preprocessing and prediction pipeline used by FastAPI
- Dockerfile – Uses public.ecr.aws/lambda/python:3.12 base and uv to install dependencies from pyproject.toml

Dockerfile overview
- FROM public.ecr.aws/lambda/python:3.12
- COPY uv binary from astral/uv image
- COPY pyproject.toml and uv.lock
- RUN uv pip install --system --requirement pyproject.toml
- COPY lambda_function.py and machine_failure_prediction.pkl
- CMD ["lambda_function.lambda_handler"]

Build the image
- docker build –platform linux/amd64 -t machine-failure-prediction .

Local invocation using the Lambda runtime interface
1. Run the container with the Lambda Runtime Interface Emulator port:
   - docker run -p 9000:8010 machine-failure-prediction:latest
2. Invoke with a sample event (API Gateway proxy-style event):
   - curl "http://localhost:9000/2015-03-31/functions/function/invocations" \
       -d '{
             "body": "{\\"Hydraulic_Pressure\\": 71.04, \\"Coolant_Pressure\\": 6.93, \\"Air_System_Pressure\\": 6.28, \\"Coolant_Temperature\\": 25.6, \\"Hydraulic_Oil_Temperature\\": 46.0, \\"Spindle_Bearing_Temperature\\": 33.4, \\"Spindle_Vibration\\": 1.29, \\"Tool_Vibration\\": 26.49, \\"Spindle_Speed\\": 25892.0, \\"Voltage\\": 335.0, \\"Torque\\": 24.06, \\"Cutting\\": 3.58 }"
          }'

Expected response format:
{
  "statusCode": 200,
  "body": "{\"prediction\": 0}"
}

Push to ECR and deploy to Lambda
1. Authenticate Docker to ECR and create a repository
2. Tag and push the image to ECR
3. Create a Lambda function using the container image
4. (Optional) Add an API Gateway HTTP API to expose the Lambda as an HTTP endpoint

Event format for Lambda
- The handler expects an API Gateway proxy-style event where the body field is a JSON string containing the same keys as the FastAPI version (simplified keys are acceptable, the handler maps to training-time names internally).


## Operational Notes and Tips

- Cold start: The Lambda loads the pickle once per container. Subsequent invocations re-use the loaded objects.
- Numeric stability: Ensure all 12 required features are provided. Missing/extra keys will result in errors.
- Scaling consistency: Do not change the order of parameters in parameters list; it must match the scaler and vectorizer training configuration.
- Model updates: When retraining, regenerate machine_failure_prediction.pkl with dv, model, and sc in the same order and overwrite the artifact in both the API and Docker image.


## Troubleshooting

- ImportError / missing packages:
  - Confirm pyproject.toml lists all required dependencies (fastapi, uvicorn, xgboost, scikit-learn, pandas, pydantic, requests, etc.)
- FastAPI receives 422 Unprocessable Entity:
  - Validate the request JSON keys and numeric types match the expected simplified keys shown above.
- Lambda returns 500 error:
  - Check CloudWatch logs for stack traces. Common errors include invalid event.body JSON or missing keys.
- Local Lambda test fails to start:
  - Ensure the container built successfully and that the CMD points to lambda_function.lambda_handler.


## License and Attribution

This project is intended for educational and demonstration purposes. Replace or add license information appropriate for your use case.
