import requests

url = 'http://127.0.0.1:5000/request'
data = {
    "prompt": "ocean",
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 0
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Square result: {result}")
else:
    print(f"Error: {response.json()}")
