import requests

mobile = {
    'battery_power':1 ,
    'clock_speed':1,
    'fc':1,
    'int_memory':1,
    'm_dep':1,
    'mobile_wt':1,
    'n_cores':1,
    'pc':1,
    'px_height':1,
    'px_width':1,
    'ram':1,
    'sc_h':1,
    'sc_w':1
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=mobile)
print(response.json())