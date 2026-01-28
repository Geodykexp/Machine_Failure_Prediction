import pickle
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import xgboost as xgb # type: ignore
import pandas as pd

count = 0

app = FastAPI(title = 'Machine_Failure_Prediction_API')

# Load dv and sc at startup
with open ('machine_failure_prediction.pkl', 'rb') as f:
    dv = pickle.load(f)
    model = pickle.load(f)
    sc = pickle.load(f)

parameters = ['Hydraulic_Pressure(bar)',
              'Coolant_Pressure(bar)',
              'Air_System_Pressure(bar)',
              'Coolant_Temperature',
              'Hydraulic_Oil_Temperature(?C)',
              'Spindle_Bearing_Temperature(?C)',
              'Spindle_Vibration(?m)',
              'Tool_Vibration(?m)',
              'Spindle_Speed(RPM)',
              'Voltage(volts)',
              'Torque(Nm)',
              'Cutting(kN)',]

class MachineData(BaseModel):                   # added
    model_config = ConfigDict(populate_by_name=True)    #added

    Hydraulic_Pressure: float = Field(alias='Hydraulic_Pressure(bar)')
    Coolant_Pressure: float = Field(alias='Coolant_Pressure(bar)')
    Air_System_Pressure: float = Field(alias='Air_System_Pressure(bar)')
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float = Field(alias='Hydraulic_Oil_Temperature(?C)')
    Spindle_Bearing_Temperature: float = Field(alias='Spindle_Bearing_Temperature(?C)')
    Spindle_Vibration: float = Field(alias='Spindle_Vibration(?m)')
    Tool_Vibration: float = Field(alias='Tool_Vibration(?m)')
    Spindle_Speed: float = Field(alias='Spindle_Speed(RPM)')
    Voltage: float = Field(alias='Voltage(volts)')
    Torque: float = Field(alias='Torque(Nm)')
    Cutting: float = Field(alias='Cutting(kN)')


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.get("/")
async def read_root():
        return {"message": "Welcome to the Machine Failure Prediction API"}

@app.post('/predict')
async def predict(data: MachineData):
    import xgboost as xgb # type: ignore
    global count
    count += 1
    print(f"Request {count}: {data}")

    try:
        data_dict = data.model_dump(by_alias=True)  #added
        parameter = {k:data_dict[k] for k in parameters}
        scaled_parameter = sc.transform(pd.DataFrame([parameter]))
        for i, k in enumerate(parameters):
            data_dict[k] = scaled_parameter[0][i]
        print(f"data_dict: {data_dict}")
        
        X = dv.transform([data_dict])
        feature_names=dv.get_feature_names_out().tolist()   
        dmat = xgb.DMatrix(X, feature_names=feature_names) 
        y_pred = float(model.predict(dmat)[0])
        return {"prediction": int(y_pred), "count":count}
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
         
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8010)
    uvicorn.run("main:app", host="127.0.0.1", port=8010, log_level="info", reload=True)    


