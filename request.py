import requests

url ='http://34.254.252.228:8080/predict/similar_images'
f = {'image': open('images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg', 'rb')}
data = {"matches":'3'}
response = requests.post(url, files=f, data= data)

print(response.json())