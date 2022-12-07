import requests

inputdata = {'age': 29,

'workclass': 'Private',

'fnlgt': 12234,

'education': 'Bachelors',

'education_num': 12,

'marital_status': 'Married_civ_spouse',

'occupation': 'Prof_specialty',

'relationship': 'Wife',

'race': 'White',

'sex': 'Female',

'capital_gain': 0,

'capital_loss': 0,

'hours_per_week': 40,

'native_country': 'Cuba'}

response = requests.post(

url='https://scalable-ml-pipe-prod.herokuapp.com/predict',

json=inputdata)

print(response.status_code)

print(response.json())