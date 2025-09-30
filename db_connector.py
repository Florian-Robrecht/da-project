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
    "accept": "application/json",
    "DB-Api-Key": os.getenv("CLIENT_SECRET"),
    "DB-Client-ID": os.getenv("CLIENT_ID"),
}

response = requests.get(url, headers=headers, params=querystring)

with open('all_stations.json', 'w') as f:
    json.dump(response.json(), f)