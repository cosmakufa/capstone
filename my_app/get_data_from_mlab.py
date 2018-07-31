import requests
import datetime
import time
import json
import decimal
from pymongo import MongoClient
import pymongo
from collections import defaultdict
import pandas as pd
import os

dbuser = os.getenv('dbuser')
dbpassword =  os.getenv('dbpassword')
dbip = os.getenv('dbip')

query = 'mongodb://' + dbuser +':' + dbpassword + '@' + dbip +'.mlab.com:25792/recipe'
client = MongoClient(query)
db = client['recipe']
collection = db['recipes']

cursor = collection.find({})
keys = set()
for entry in cursor:
    for key in entry:
        keys.add(key)

cursor = collection.find({})
result = defaultdict(list)
for entry in cursor:
    for key in keys: 
        result[key].append(entry.get(key, None))

df = pd.DataFrame.from_dict(result)
df.to_csv('recipe.csv')



