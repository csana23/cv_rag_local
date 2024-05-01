import requests

response = requests.post("http://localhost:8000/query-model", json={"text": "What is the candidate's education level?"})

print(response)