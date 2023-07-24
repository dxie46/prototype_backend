import requests
from PIL import Image

response = requests.post("http://localhost:5000/eye-pressure/predict", files={'image': open('./test_image.png', 'rb')})
print(response.json())