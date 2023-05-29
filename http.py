import bentoml
import json
import requests
import numpy as np

input_data = np.array([1, 2, 3])
headers = {"Content-Type": "application/json"}
response = requests.post("http://47.107.162.21:5002/predict", headers=headers, data=json.dumps({"input_series": input_data.tolist()}))
output_data = np.array(json.loads(response.text)["data"])