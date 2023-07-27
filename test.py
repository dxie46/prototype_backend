import requests

url = 'http://localhost:5000/eye-pressure/predict'
response = requests.post(url, files={'image': open('./test_image.png', 'rb')})
print(response)