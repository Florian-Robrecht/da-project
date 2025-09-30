"""
A script connecting to the Deutsche Bahn API via the requests library.

Retrieving location data for train stations within Germany. 
"""
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://apis.deutschebahn.com/db-api-marketplace/apis/station-data/v2/stations"

querystring = {"limit": "10000", "category": "1-7"}

headers = {
    #"cookie": "TS0165d0f4=01d513bcd1dc157dd2fe33fbc17e2d7e409a282b140a7a38f5ed7ed1b7bfd1b64464ec5519d71556bd480788d38d898321ba549fb1",
    "accept": "application/json",
    "DB-Api-Key": os.getenv("CLIENT_SECRET"),
    "DB-Client-ID": os.getenv("CLIENT_ID"),
}

response = requests.get(url, headers=headers, params=querystring)

with open('all_stations.json', 'w') as f:
    json.dump(response.json(), f)