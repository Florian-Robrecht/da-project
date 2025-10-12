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

querystring = {"limit": "10000", "category": "1-2"}

headers = {
    "accept": "application/json",
    "DB-Api-Key": os.getenv("CLIENT_SECRET"),
    "DB-Client-ID": os.getenv("CLIENT_ID"),
}

response = requests.get(url, headers=headers, params=querystring)

response = response.json()

final_station_data = []
for station in response["result"]:
    print(station["name"])
    final_station_data.append(
        {
            "name": station["name"],
            "address": f'{station["mailingAddress"]["street"]}, {station["mailingAddress"]["zipcode"]} {station["mailingAddress"]["city"]}',
            "lat": (
                station["evaNumbers"][0]["geographicCoordinates"]["coordinates"][1]
                if len(station["evaNumbers"]) > 0
                else None
            ),
            "lon": (
                station["evaNumbers"][0]["geographicCoordinates"]["coordinates"][0]
                if len(station["evaNumbers"]) > 0
                else None
            ),
        }
    )


with open("all_stations_regio_ice_1_2.json", "w") as f:
    json.dump(final_station_data, f)
