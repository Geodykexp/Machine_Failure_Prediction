import pickle
import xgboost as xgb # type: ignore
import pandas as pd
import pandas as pd

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

client = {
    'Hydraulic_Pressure(bar)': 71.04,
    'Coolant_Pressure(bar)': 6.93,
    'Air_System_Pressure(bar)': 6.28,
    'Coolant_Temperature': 25.6,
    'Hydraulic_Oil_Temperature(?C)': 46.0,
    'Spindle_Bearing_Temperature(?C)': 33.4,
    'Spindle_Vibration(?m)': 1.29,
    'Tool_Vibration(?m)': 26.49,
    'Spindle_Speed(RPM)': 25892.0,
    'Voltage(volts)': 335.0,
    'Torque(Nm)': 24.06,
    'Cutting(kN)': 3.58
}


global count
count = 0
count += 1

try:
    data_dict = client
    parameter = {k:data_dict[k] for k in parameters}
    scaled_parameter = sc.transform(pd.DataFrame([parameter]))
    for i, k in enumerate(parameters):
        data_dict[k] = scaled_parameter[0][i]
    print(f"data_dict: {data_dict}")

    X = dv.transform([data_dict])
    print("X shape:", X.shape)
    feature_names=dv.get_feature_names_out().tolist()   
    dmat = xgb.DMatrix(X, feature_names=feature_names)  
    y_pred = float(model.predict(dmat)[0]) 
    print({"prediction": int(y_pred), "count":count})
except Exception as e:
    print(f"Error in prediction: {e}")    
   