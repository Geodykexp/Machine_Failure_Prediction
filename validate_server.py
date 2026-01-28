import requests

url = "http://127.0.0.1:8010/predict"

client = {
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
try:
    response = requests.post(url, json=client)
    response.raise_for_status()
    data = response.json()
    print(data)
except requests.exceptions.RequestException as e:
    print(f"Error during request: {e}")


