"""
A Streamlit web application to visualize train station accessibility in Germany.

This app displays a pre-computed isochrone map showing travel times to the
nearest train station. Users can input an address to place a marker on the map.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- CONFIGURATION ---
CONFIG = {
    "GEOJSON_PATH": "train_station_access.geojson",
    "APP_TITLE": "🚄 Train Station Accessibility in Germany",
    "INITIAL_MAP_LOCATION": [51.1657, 10.4515],  # Centered on Germany
    "INITIAL_MAP_ZOOM": 6,
}

# --- HELPER FUNCTIONS ---


@st.cache_data
def load_geojson_data(path: str):
    """
    Loads the pre-computed GeoJSON data from the given path.
    Uses Streamlit's caching to avoid reloading the file on every interaction.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(
            f"Error: The file '{path}' was not found. Please ensure it is in the correct directory."
        )
        return None


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


# --- MAIN APPLICATION LOGIC ---


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
    - **Green:** < 10 minutes
    - **Orange:** 10-20 minutes
    - **Red:** > 20 minutes (not shown in this demo)
    """
    )

    # --- USER INPUT ---
    address_input = st.text_input(
        "Enter a German address to locate on the map:",
        placeholder="e.g., Adickesalle 32-34",
    )

    # --- MAP DISPLAY ---
    # Load the pre-computed accessibility data
    geojson_data = load_geojson_data(CONFIG["GEOJSON_PATH"])

    if geojson_data:
        # Initialize map centered on Germany or the last searched location
        if "center" not in st.session_state:
            st.session_state.center = CONFIG["INITIAL_MAP_LOCATION"]
            st.session_state.zoom = CONFIG["INITIAL_MAP_ZOOM"]

        # Geocode the address when input is provided
        if address_input:
            location_coords = geocode_address(address_input)
            if location_coords:
                st.session_state.center = location_coords
                st.session_state.zoom = 13  # Zoom in closer on a specific address
                st.success(f"Found location: {location_coords}")
            else:
                st.error("Could not find the address. Please try another one.")

        # Create the Folium map object
        m = folium.Map(
            location=st.session_state.center,
            zoom_start=st.session_state.zoom,
            tiles="cartodbpositron",  # A clean, neutral basemap
        )

        # Add the accessibility layer
        folium.GeoJson(
            geojson_data, style_function=style_function, name="train_station Accessibility"
        ).add_to(m)

        # Add a marker for the searched location
        if (
            "center" in st.session_state
            and st.session_state.zoom > CONFIG["INITIAL_MAP_ZOOM"]
        ):
            folium.Marker(
                location=st.session_state.center,
                popup=address_input,
                icon=folium.Icon(color="blue", icon="home"),
            ).add_to(m)

        # Display the map in the Streamlit app
        st_folium(
            m,
            width="100%",
            height=1000,
            center=st.session_state.center,
            zoom=st.session_state.zoom,
        )


if __name__ == "__main__":
    main()
