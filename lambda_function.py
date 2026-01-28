import pickle
import json
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load model at cold start
with open('machine_failure_prediction.pkl', 'rb') as f:
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

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])  # Assuming API Gateway event
        parameter = {k: data[k] for k in parameters}
        scaled_parameter = sc.transform(pd.DataFrame([parameter]))
        for i, k in enumerate(parameters):
            data[k] = scaled_parameter[0][i]
        X = dv.transform([data])
        dmat = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
        y_pred = int(model.predict(dmat)[0])
        return {'statusCode': 200, 'body': json.dumps({'prediction': y_pred})}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    

   