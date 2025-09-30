"""
A Streamlit web application to visualize train station accessibility in Germany.

This app displays a pre-computed isochrone map showing travel times to the
nearest train station. Users can input an address to place a marker on the map.
"""

import copy
import json
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- CONFIGURATION ---
CONFIG = {
    "STATIONS_JSON_PATH": "all_stations.json",
    "APP_TITLE": "🚄 Train Station Accessibility in Germany",
    "INITIAL_MAP_LOCATION": [51.1657, 10.4515],  # Centered on Germany
    "INITIAL_MAP_ZOOM": 6,
}

# --- HELPER FUNCTIONS ---


@st.cache_data
def load_stations_data(path: str):
    """Loads train station data from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode the JSON file.")
        return None


def add_station_markers_to_map(folium_map, stations_data):
    """
    Adds station markers to a Folium map using a MarkerCluster for performance.
    """
    if not stations_data:
        return

    marker_cluster = MarkerCluster().add_to(folium_map)
    icon = folium.CustomIcon(
        "https://marketingportal.extranet.deutschebahn.com/resource/blob/9686952/4a63e275e78190f96e64aa48bbde6c63/Start-favicon.ico?v=1",
        icon_size=(38, 38),
    )

    # Loop through the station data and add markers to the cluster
    for station in stations_data:
        try:
            name = station.get("name", "N/A")
            address = station.get("address", "")
            lat = station.get("lat")
            lon = station.get("long")

            # Ensure coordinates are valid
            if lat is not None and lon is not None:
                popup_text = f"<b>{name}</b><br>{address}"
                folium.Marker(
                    location=[lat, lon], popup=popup_text, tooltip=name, icon=icon
                ).add_to(marker_cluster)
        except (KeyError, TypeError) as e:
            st.warning(f"Skipping a station due to invalid data: {e}")



def create_folium_map():
    """
    Creates and caches the Folium map with all station markers.
    This function will only run once.
    """

    stations_data = load_stations_data(CONFIG["STATIONS_JSON_PATH"])

    m = folium.Map(
        location=CONFIG["INITIAL_MAP_LOCATION"],
        zoom_start=CONFIG["INITIAL_MAP_ZOOM"],
        tiles="cartodbpositron",
    )

    if stations_data:
        add_station_markers_to_map(m, stations_data)

    return m


def geocode_address(address: str):
    """
    Converts a string address into latitude and longitude coordinates.
    Returns a tuple (lat, lon) or None if not found.
    """
    if not address:
        return None

    geolocator = Nominatim(user_agent="train_station_accessibility_app")
    try:
        location = geolocator.geocode(address, country_codes="DE")
        if location:
            return (location.latitude, location.longitude)
        return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.warning("Geocoding service is unavailable. Please try again later.")
        return None
def style_function(feature):
    """
    Defines the color styling for the GeoJSON layer based on its properties.
    """
    return {
        "fillColor": feature["properties"]["color"],
        "color": feature["properties"]["color"],
        "weight": 1,
        "fillOpacity": 0.5,
    }


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title=CONFIG["APP_TITLE"], layout="wide")
    st.title(CONFIG["APP_TITLE"])

    st.markdown(
        """
    This map shows the estimated walking time to the nearest train station in Germany.
    Enter an address below to see where it falls on the map.
    """
    )

    #--- USER INPUT ---
    address_input = st.text_input(
        "Enter a German address to locate on the map:",
        placeholder="e.g., Brandenburger Tor, Berlin",
    )

    # --- MAP DISPLAY ---
    # Get the base map from the cache. This is always fast.
    base_map = create_folium_map()

    # Initialize session state for map view and the map object itself
    if "center" not in st.session_state:
        st.session_state.center = CONFIG["INITIAL_MAP_LOCATION"]
        st.session_state.zoom = CONFIG["INITIAL_MAP_ZOOM"]
        # Store the base map as the initial map to display
        st.session_state.map_to_display = base_map

    
    if address_input:
        location_coords = geocode_address(address_input)
        if location_coords:
            st.session_state.center = location_coords
            st.session_state.zoom = 13

            # Create a deep copy and add the marker
            map_with_marker = copy.deepcopy(base_map)
            folium.Marker(
                location=st.session_state.center,
                popup=address_input,
                icon=folium.Icon(color="blue", icon="home"),
            ).add_to(map_with_marker)

            # Overwrite the map in session state with the new one
            st.session_state.map_to_display = map_with_marker
            st.success(f"Found location: {location_coords}")
        else:
            st.error("Could not find the address. Please try another one.")
    # On every run (including pans/zooms), display the map from session state.
    # This avoids the deepcopy on simple interactions.
    st_folium(
        st.session_state.map_to_display,
        width="100%",
        height=1000,
        center=st.session_state.center,
        zoom=st.session_state.zoom,
    )


if __name__ == "__main__":
    main()
