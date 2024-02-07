import requests

response = requests.get('https://dummy.restapiexample.com/api/v1/employees')
print(response.status_code)
print(response.text)


response = requests.post('https://dummy.restapiexample.com/api/v1/create', data={"name":"test","salary":"123","age":"23"})
print(response.status_code)
print(response.text)
