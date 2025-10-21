"""
A Streamlit web application to visualize train station accessibility in Germany using Pydeck.

This app displays all train stations on an interactive map. Users can input
an address to place a marker and zoom to its location.
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
import json
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- CONFIGURATION ---
CONFIG = {
    "STATIONS_JSON_PATH": "datasets/ev_stations.json",
    "APP_TITLE": "🔋🚗🇩🇪 Public EV Charging Stations in Germany",
    "INITIAL_MAP_LOCATION": [51.1657, 10.4515],
    "INITIAL_MAP_ZOOM": 5,
    "MAPBOX_STYLE": "mapbox://styles/mapbox/light-v9",
    "EV_CHARGER_GREEN_ICON_URL": "https://cdn.prod.website-files.com/672b9491a2b3a49f453f8338/68eb959c10a234c6efe509fd_1.png",
    "EV_CHARGER_YELLOW_ICON_URL": "https://cdn.prod.website-files.com/672b9491a2b3a49f453f8338/68eb959c530c133f73e18dd0_2.png",
    "EV_CHARGER_RED_ICON_URL": "https://cdn.prod.website-files.com/672b9491a2b3a49f453f8338/68eb959ccac479937c34e3b6_3.png",
    "EV_CHARGER_DATASET_PUBLICATION": datetime(2025, 8, 26),
}
REF_YEAR = CONFIG["EV_CHARGER_DATASET_PUBLICATION"].year
REF_MONTH = CONFIG["EV_CHARGER_DATASET_PUBLICATION"].month

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_and_prepare_data(path: str) -> pd.DataFrame | None:
    """Loads EV charging station data and prepares it for Pydeck."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df.dropna(subset=["lat", "lon"], inplace=True)

        # 1. Define the distinct icon data dictionaries
        # Using .copy() is crucial so each dictionary is a separate object
        base_icon_data = {"width": 128, "height": 128, "anchorY": 128}

        icon_data_red = base_icon_data.copy()
        icon_data_red["url"] = CONFIG["EV_CHARGER_RED_ICON_URL"]

        icon_data_yellow = base_icon_data.copy()
        icon_data_yellow["url"] = CONFIG["EV_CHARGER_YELLOW_ICON_URL"]

        icon_data_green = base_icon_data.copy()
        icon_data_green["url"] = CONFIG["EV_CHARGER_GREEN_ICON_URL"]

        def get_icon(power_kw: float) -> dict:
            """Returns the correct icon dictionary based on charging power."""
            if power_kw < 50:
                return icon_data_red
            elif 50 <= power_kw < 150:
                return icon_data_yellow
            else:  # >= 150 kW
                return icon_data_green

        power_col = "Nennleistung Ladeeinrichtung [kW]"
        df[power_col] = pd.to_numeric(df[power_col], errors="coerce")

        df["icon_data"] = df[power_col].apply(get_icon)
        return df

    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error processing the data file: {e}")
        return None


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
        f"""
        This website shows the progress of the transition to EVs in Germany - updated on {CONFIG["EV_CHARGER_DATASET_PUBLICATION"].strftime('%Y-%m-%d')}.
        """
    )
    # --- LOAD DATA ---
    ev_station_df = load_and_prepare_data(CONFIG["STATIONS_JSON_PATH"])
    if ev_station_df is not None and not ev_station_df.empty:
        ev_station_df["Inbetriebnahmedatum"] = pd.to_datetime(
            ev_station_df["Inbetriebnahmedatum"], format="%d.%m.%Y"
        )
    ev_registrations_df = pd.read_csv("datasets/ev_registrations_cleaned.csv")
    
    # Initialize filtered dataframe
    ev_station_df_filtered = ev_station_df
    selected_year = None
    

    # --- CREATE INFO CARDS ---
    # column1: EV Stations, column2: Registered EVs
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            num_chargers = ev_station_df_filtered["Anzahl Ladepunkte"].sum()
            # EV_CHARGER_DATASET_PUBLICATION - 1 year
            one_year_ago = CONFIG["EV_CHARGER_DATASET_PUBLICATION"].replace(
                year=CONFIG["EV_CHARGER_DATASET_PUBLICATION"].year - 1
            )
            num_chargers_one_year_ago = ev_station_df[
                ev_station_df["Inbetriebnahmedatum"] <= one_year_ago
            ]["Anzahl Ladepunkte"].sum()
            
            # Show different label based on whether filtering is applied
            if selected_year and selected_year < max_year:
                label = f"🔋 Public EV Chargers in Germany (up to {selected_year})"
            else:
                label = "🔋 Public EV Chargers in Germany"
                
            st.metric(
                label=label,
                value=f"{num_chargers:,}",
                delta=f"{round(((num_chargers/num_chargers_one_year_ago)-1)*100,2)}% YoY",
            )
        except (FileNotFoundError, ValueError):
            st.metric("🔋 Public EV Chargers in Germany", "Data not available")
    with col2:
        try:
            registered_evs = ev_registrations_df["Count"].sum()
            registered_evs_one_year_ago = ev_registrations_df[
                (
                    (ev_registrations_df["Year"] < REF_YEAR - 1)
                    | (
                        (ev_registrations_df["Year"] == REF_YEAR - 1)
                        & (ev_registrations_df["Month"] <= REF_MONTH)
                    )
                )
            ]["Count"].sum()
            st.metric(
                label="🚗 Registered EVs & Plug-In hybrids",
                value=f"{registered_evs:,}",
                delta=f"{round(((registered_evs/registered_evs_one_year_ago)-1)*100,2)}% YoY",
            )
        except (FileNotFoundError, pd.errors.ParserError):
            st.metric("🚗 Registered EVs", "Data not available")
    with col3:
        try:
            st.metric(
                label="🔌 Registered EVs per public charger",
                value=round(registered_evs / num_chargers, 2),
                delta=f"{round(((registered_evs/num_chargers)/(registered_evs_one_year_ago/num_chargers_one_year_ago)-1)*100,2)}% YoY",
            )
        except (FileNotFoundError, pd.errors.ParserError):
            st.metric("📍 Charging Stations", "Data not available")

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🔍 Map Controls")
        
        # --- ADDRESS SEARCH ---
        st.subheader("📍 Address Search")
        address_input = st.text_input(
            "Enter a German address:",
            placeholder="Adickesallee 32",
            help="Search for a specific location on the map"
        )
        
        # Handle address geocoding
        coords = None
        if address_input:
            coords = geocode_address(address_input)
            if coords:
                st.success(f"📍 Found: {coords[0]:.5f}, {coords[1]:.5f}")
            else:
                st.error("Address not found. Please try another one.")
        
        st.divider()
        
        # --- POWER FILTER ---
        st.subheader("⚡ Power Filter")
        
        # Power level filter checkboxes
        show_low_power = st.checkbox("Chargers < 50kW", value=True, help="Show low power chargers")
        show_medium_power = st.checkbox("Chargers ≥ 50kW and < 150kW", value=True, help="Show medium power chargers")
        show_high_power = st.checkbox("Chargers ≥ 150kW", value=True, help="Show high power chargers")
        
        # Show power distribution info
        if ev_station_df is not None and not ev_station_df.empty:
            power_col = "Nennleistung Ladeeinrichtung [kW]"
            
            # Count stations by power level
            low_power_count = len(ev_station_df[ev_station_df[power_col] < 50])
            medium_power_count = len(ev_station_df[(ev_station_df[power_col] >= 50) & (ev_station_df[power_col] < 150)])
            high_power_count = len(ev_station_df[ev_station_df[power_col] >= 150])
            
            st.caption(f"""
            **Power Distribution:**
            - < 50kW: {low_power_count:,} stations
            - 50-150kW: {medium_power_count:,} stations  
            - ≥ 150kW: {high_power_count:,} stations
            """)
        
        st.divider()
        
        # --- YEAR FILTER ---
        st.subheader("📅 Time Filter")
        
        if ev_station_df is not None and not ev_station_df.empty:
            # Extract year range from the data
            min_year = int(ev_station_df["Inbetriebnahmedatum"].dt.year.min())
            max_year = int(ev_station_df["Inbetriebnahmedatum"].dt.year.max())
            
            # Create year slider
            selected_year = st.slider(
                "Show stations up to:",
                min_value=min_year,
                max_value=max_year,
                value=max_year,  # Default to showing all stations
                step=1,
                help="Move the slider to see how the charging station network developed over time"
            )
            
            # Filter data based on selected year
            ev_station_df_filtered = ev_station_df[
                ev_station_df["Inbetriebnahmedatum"].dt.year <= selected_year
            ].copy()
            
            # Show information about filtered data
            total_stations = len(ev_station_df)
            filtered_stations = len(ev_station_df_filtered)
            if selected_year < max_year:
                st.caption(f"Showing {filtered_stations:,} stations (out of {total_stations:,} total) up to {selected_year}")
            else:
                st.caption(f"Showing all {filtered_stations:,} stations")
        else:
            st.info("No charging station data available")
            selected_year = None
            ev_station_df_filtered = ev_station_df
    
    # --- APPLY POWER FILTERING ---
    if ev_station_df_filtered is not None and not ev_station_df_filtered.empty:
        power_col = "Nennleistung Ladeeinrichtung [kW]"
        power_mask = pd.Series([False] * len(ev_station_df_filtered), index=ev_station_df_filtered.index)
        
        # Apply power filters
        if show_low_power:
            power_mask |= (ev_station_df_filtered[power_col] < 50)
        if show_medium_power:
            power_mask |= ((ev_station_df_filtered[power_col] >= 50) & (ev_station_df_filtered[power_col] < 150))
        if show_high_power:
            power_mask |= (ev_station_df_filtered[power_col] >= 150)
        
        # Apply the power filter
        ev_station_df_filtered = ev_station_df_filtered[power_mask].copy()
        
        # Show power filter info in sidebar
        with st.sidebar:
            if not (show_low_power and show_medium_power and show_high_power):
                filtered_power_stations = len(ev_station_df_filtered)
                st.caption(f"⚡ Power filter: {filtered_power_stations:,} stations visible")

    # --- INITIALIZE MAP STATE ---
    # Use session_state to preserve the map's view across reruns
    if "view_state" not in st.session_state:
        st.session_state.view_state = pdk.ViewState(
            latitude=CONFIG["INITIAL_MAP_LOCATION"][0],
            longitude=CONFIG["INITIAL_MAP_LOCATION"][1],
            zoom=CONFIG["INITIAL_MAP_ZOOM"],
            pitch=0,
        )

    # Update map view if address was found
    if coords:
        st.session_state.view_state.latitude = coords[0]
        st.session_state.view_state.longitude = coords[1]
        st.session_state.view_state.zoom = 10

    # List to hold all pydeck layers
    layers = []

    # --- CREATE STATION LAYER ---
    if ev_station_df_filtered is not None and not ev_station_df_filtered.empty:
        ev_station_layer_green = pdk.Layer(
            "IconLayer",
            data=ev_station_df_filtered,
            get_icon="icon_data",
            get_position="[lon, lat]",
            size_units="pixels",
            get_size=15,
            size_scale=1,
            size_min_pixels=5,
            pickable=True,
        )
        layers.append(ev_station_layer_green)

    # --- CREATE USER MARKER ---
    user_location_df = None
    if coords:
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
    # Configure tooltip to show EV station details on hover
    tooltip = {
        "html": "<b>{Betreiber}</b><br/>{Nennleistung Ladeeinrichtung [kW]}kW </b><br/> Chargers: {Anzahl Ladepunkte}",
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
