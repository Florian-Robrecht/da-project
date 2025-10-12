"""
A Streamlit web application to visualize train station accessibility in Germany using Pydeck.

This app displays all train stations on an interactive map. Users can input
an address to place a marker and zoom to its location.
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
import json
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- CONFIGURATION ---
CONFIG = {
    "STATIONS_JSON_PATH": "all_stations.json",
    "APP_TITLE": "🚄 Train Station Accessibility in Germany",
    "INITIAL_MAP_LOCATION": [51.1657, 10.4515],
    "INITIAL_MAP_ZOOM": 5,
    "MAPBOX_STYLE": "mapbox://styles/mapbox/light-v9",
    "DB_ICON_URL": "https://cdn.prod.website-files.com/672b9491a2b3a49f453f8338/68dbebf8274a54aae784d3dd_db_logo.png",
}

# --- HELPER FUNCTIONS ---


@st.cache_data
def load_and_prepare_data(path: str) -> pd.DataFrame | None:
    """Loads train station data and prepares it for Pydeck."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        # Define the icon data once
        icon_data = {
            "url": CONFIG["DB_ICON_URL"],
            "width": 128,
            "height": 128,
            "anchorY": 128,
        }
        # Add the icon data as a new column to every row in the DataFrame
        df["icon_data"] = [icon_data] * len(df)
        # Drop rows with missing coordinates
        df.dropna(subset=["lat", "lon"], inplace=True)
        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error processing the data file: {e}")
        return None


#@st.cache_data
#def load_grid_data(path: str) -> pd.DataFrame | None:
#    """Loads the pre-computed grid data from a JSON file."""
#    try:
#        df = pd.read_json(path)
#        return df
#    except FileNotFoundError:
#        st.error(f"Error: The grid data file '{path}' was not found.")
#        st.info("Please run the `generate_grid.py` script first to create it.")
#        return None


def geocode_address(address: str) -> tuple[float, float] | None:
    """
    Converts a string address into latitude and longitude coordinates.
    Returns a tuple (lat, lon) or None if not found.
    """
    if not address:
        return None

    geolocator = Nominatim(user_agent="train_station_accessibility_app_pydeck")
    try:
        location = geolocator.geocode(address, country_codes="DE", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.warning("Geocoding service is unavailable. Please try again later.")
        return None


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title=CONFIG["APP_TITLE"], layout="wide")
    st.title(CONFIG["APP_TITLE"])

    st.markdown(
        """
        This map shows train station locations and estimates walking accessibility in Germany. 
        The colored overlay represents the air-distance to the nearest station:
        - **Green**: < 1 km
        - **Yellow**: 1 - 2.5 km
        - **Red**: > 2.5 km
        """
    )

    # --- LOAD DATA ---
    station_df = load_and_prepare_data(CONFIG["STATIONS_JSON_PATH"])
    #grid_df = load_grid_data("reachability_grid.json")
    # --- USER INPUT ---
    address_input = st.text_input(
        "Enter a German address to locate on the map:",
        placeholder="e.g., Brandenburger Tor, Berlin",
    )

    # --- INITIALIZE MAP STATE ---
    # Use session_state to preserve the map's view across reruns
    if "view_state" not in st.session_state:
        st.session_state.view_state = pdk.ViewState(
            latitude=CONFIG["INITIAL_MAP_LOCATION"][0],
            longitude=CONFIG["INITIAL_MAP_LOCATION"][1],
            zoom=CONFIG["INITIAL_MAP_ZOOM"],
            pitch=0,
        )

    # List to hold all pydeck layers
    layers = []

    # --- CREATE STATION LAYER ---
    if station_df is not None and not station_df.empty:
        station_layer = pdk.Layer(
            "IconLayer",
            data=station_df,
            get_icon="icon_data",
            get_position="[lon, lat]",
            size_units="meters",  # Set the size unit to meters
            get_size=750,  # Each icon will represent a 750-meter space
            size_scale=1,
            pickable=True,
        )
        layers.append(station_layer)

    # --- HANDLE ADDRESS GEOCODING AND CREATE USER MARKER ---
    user_location_df = None
    if address_input:
        coords = geocode_address(address_input)
        if coords:
            st.success(
                f"📍 Found location: Latitude={coords[0]:.5f}, Longitude={coords[1]:.5f}"
            )
            # Update map view to center on the new address
            st.session_state.view_state.latitude = coords[0]
            st.session_state.view_state.longitude = coords[1]
            st.session_state.view_state.zoom = 10

            # Create a DataFrame for the user's location marker
            user_location_df = pd.DataFrame(
                [
                    {
                        "name": "Your Searched Location",
                        "address": address_input,
                        "lat": coords[0],
                        "lon": coords[1],
                    }
                ]
            )

        else:
            st.error("Could not find the address. Please try another one.")

    if user_location_df is not None:
        user_layer = pdk.Layer(
            "ScatterplotLayer",
            data=user_location_df,
            get_position="[lon, lat]",
            get_color="[128, 0, 128, 255]",  # Red-orange color
            get_radius=100,  # Radius in meters
            pickable=True,
            radius_min_pixels=5,
        )
        layers.append(user_layer)

    # --- RENDER MAP ---
    # Configure tooltip to show station details on hover
    tooltip = {
        "html": "<b>{name}</b><br/>{address}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "font-family": "sans-serif",
        },
    }

    # Access Mapbox API key from Streamlit secrets
    mapbox_api_key = st.secrets.get("MAPBOX_API_KEY")

    st.pydeck_chart(
        pdk.Deck(
            map_style=CONFIG["MAPBOX_STYLE"],
            initial_view_state=st.session_state.view_state,
            layers=layers,
            tooltip=tooltip,
            api_keys={"mapbox": mapbox_api_key},
        ),
        use_container_width=True,
        height=700,
    )


if __name__ == "__main__":
    main()
