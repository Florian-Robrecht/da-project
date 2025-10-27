"""This file is used to clean the dataset provided by KBA (Kraftfahrtbundesamt)"""

import pandas as pd
import json

MONTH_MAP = {
    "Januar": 1,
    "Februar": 2,
    "März": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Dezember": 12,
}


def clean_ev_charging_station_data():
    """Cleans the EV charging station data from the CSV file.
    Args:
        None

    Returns:
        None
    """
    df = pd.read_csv(
        "datasets/Ladesaeulenregister_BNetzA_2025-08-26.csv",
        sep=";",
        encoding="latin1",
        skiprows=10,
        decimal=",",
    )

    # only where Status ="In Betrieb"
    df = df[df["Status"] == "In Betrieb"]

    # rename "Breitengrad","Längengrad" to lat and lon columns
    df.rename(columns={"Breitengrad": "lat", "Längengrad": "lon"}, inplace=True)

    columns_to_use = [
        "Betreiber",
        "Nennleistung Ladeeinrichtung [kW]",
        "Postleitzahl",
        "Ort",
        "Straße",
        "Hausnummer",
        "lat",
        "lon",
        "Anzahl Ladepunkte",
        "Inbetriebnahmedatum",
    ]

    df[columns_to_use].to_json(
        "datasets/ev_stations.json",
        orient="records",
        indent=4,
        force_ascii=False, 
    )


def clean_ev_registration_data():
    """Cleans the EV registration data from the JSON file.
    Args:
        None

    Returns:
        None
    """
    with open("datasets/ev_registrations.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    date_parts = df["datum"].str.split(n=1, expand=True)

    # Create 'Year' column from the second part and convert it to an integer
    df["Year"] = date_parts[1].astype(int)

    # Create 'Month' column by mapping the first part using our dictionary
    df["Month"] = date_parts[0].map(MONTH_MAP)

    # 3. Rename the 'model' column to 'Type' and 'count' to 'Count'
    df.rename(columns={"model": "Type", "count": "Count"}, inplace=True)

    # 4. Select and reorder columns to create the final DataFrame
    final_df = df[["Year", "Month", "Count", "Type"]]
    final_df.to_csv("datasets/ev_registrations_cleaned.csv", index=False)

clean_ev_registration_data()