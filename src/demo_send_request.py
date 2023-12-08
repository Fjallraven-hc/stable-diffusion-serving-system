import requests

url = 'http://127.0.0.1:8080/inference'
data = {
    "prompt": "ocean",
    "height": 512,
    "width": 512,
    "loop_num": {
        "UNetModule": 50
    },
    "guidance_scale": 7.5,
    "seed": 0,
    "SLO": 10,
    "loop_index": {
        "UNetModule": 0
    }
}

response = requests.post(url, json=data)
result = response.json()
for key in result.keys():
    if type(result[key]) == float:
        print(f"key: {key}, value: {result[key]}")

"""if response.status_code == 200:
    result = response.json()
    print(f"Result: {result}")
else:
    print(f"Error: {response.json()}")
"""