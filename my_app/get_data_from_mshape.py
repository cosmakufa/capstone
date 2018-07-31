import requests
import boto3
import datetime
import time
import json
import decimal
from pymongo import MongoClient
import pymongo
import os

dbuser = os.getenv('dbuser')
dbpassword =  os.getenv('dbpassword')
dbip = os.getenv('dbip')
mshapekey = os.getenv('Mashapekey')
query = 'mongodb://' + dbuser +':' + dbpassword + '@' + dbip +'.mlab.com:25792/recipe'
client = MongoClient(query)
db = client['recipe']
collection = db['recipes']
# These code snippets use an open-source library. http://unirest.io/python
def get_data(i):
    response = requests.get("https://spoonacular-recipe-food-nutrition-v1.p.mashape.com/recipes/"+str(i)+"/information?includeNutrition=true",
  headers={
    "X-Mashape-Key": mshapekey,
    "Accept": "application/json"
  }).json()
    return response

#get_data(1)

with open('num.txt') as f:
    num = int(f.readline())

for i in range(num, num+500):
    response = get_data(i)
    
    if  "failure" != response.get("status", "success"):
        response['_id'] = response['id']
        collection.insert(response)
        with open('num.txt', 'w') as f:
            f.write(str(i+1))

'''
while (True):
    response = requests.post(url, json={'api_key': api_key,
                                    'sequence_number': sequence_number})
    raw_data = response.json()
    for entry in raw_data['data']:
        print(collection)
        entry['_next_sequence_number'] = raw_data['_next_sequence_number']
        collection.insert(entry)
    

'''

