import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "default",
    "messages": [{"role": "user", "content": "Please describe Jiangsu Nantong."}],
}

response = requests.post(url, json=data)
print(response.json())