"""
A Streamlit web application to visualize EV charging infrastructure in Germany using Pydeck.

This app displays all public EV charging stations on an interactive map. Users can input
an address to place a marker and zoom to its location. Additionally, the app displays the TEN-T core road network of Germany and the coverage
of the charging infrastructure within the TEN-T network.
"""

import json
import math
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from scipy.stats import linregress
from shapely.geometry import LineString
from shapely.ops import substring

load_dotenv()

# --- CONFIGURATION ---
CONFIG = {
    "STATIONS_JSON_PATH": "datasets/ev_stations.json",
    "ELECTRIC_CAR_REGISTRATIONS_PATH": "datasets/ev_registrations_cleaned.csv",
    "TEN_T_CORE_GEOJSON_PATH": "datasets/ten-t.geojson",
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
    """Loads EV charging station data from a JSON file and prepares it for Pydeck visualization.

    Args:
        path (str): The file path to the JSON data source.

    Returns:
        pd.DataFrame | None: A DataFrame with an 'icon_data' column ready for
        Pydeck, or None if loading or processing fails.
    """
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
            """Returns the correct icon dictionary based on charging power.
            Args:
                power_kw (float): The charging power in kilowatts.

            Returns:
                dict: The icon dictionary for the given charging power.
            """
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

    Args:
        address (str): The address to geocode.

    Returns:
        tuple[float, float] | None: A tuple (lat, lon) or None if not found.
    """
    if not address:
        return None

    geolocator = Nominatim(user_agent="ev_station_accessibility_app_pydeck")
    try:
        location = geolocator.geocode(address, country_codes="DE", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.warning("Geocoding service is unavailable. Please try again later.")
        return None


@st.cache_data
def load_tent_geojson(path: str) -> dict | None:
    """Loads the TEN-T core network GeoJSON.

    Args:
        path (str): The file path to the TEN-T GeoJSON data source.

    Returns:
        dict | None: A dictionary containing the TEN-T GeoJSON data or None if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"TEN-T file not found at: {path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Invalid TEN-T GeoJSON: {e}")
        return None


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in kilometers.

    Args:
        lon1 (float): The longitude of the first point.
        lat1 (float): The latitude of the first point.
        lon2 (float): The longitude of the second point.
        lat2 (float): The latitude of the second point.

    Returns:
        float: The great circle distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


@st.cache_data
def process_tent_roads(geojson_data: dict, segment_length_km: float = 2.5) -> list:
    """
    Process TEN-T GeoJSON into segments for coverage analysis.

    Args:
        geojson_data (dict): The TEN-T GeoJSON data
        segment_length_km: Target length for each segment in kilometers (default: 2.5km)

    Returns:
        List of segment dictionaries with start/end coordinates and midpoint
    """
    segments = []

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "LineString":
            continue

        coords = geometry.get("coordinates", [])
        if len(coords) < 2:
            continue

        # Create shapely LineString
        line = LineString(coords)

        # Sample points along the line at regular intervals
        # Approximate: 1 degree ≈ 111 km at equator, less at higher latitudes
        # For Germany (lat ~50°), 1 degree lon ≈ 71 km, 1 degree lat ≈ 111 km
        # We'll use actual distance calculation

        total_length_degrees = line.length
        # Rough approximation: convert degrees to km (using ~100 km per degree as average)
        approx_length_km = total_length_degrees * 100

        # Number of segments
        num_segments = max(int(approx_length_km / segment_length_km), 1)

        # Create segments
        for i in range(num_segments):
            start_fraction = i / num_segments
            end_fraction = (i + 1) / num_segments

            # Get segment
            segment_line = substring(
                line, start_fraction, end_fraction, normalized=True
            )

            # Get start and end coordinates
            segment_coords = list(segment_line.coords)
            if len(segment_coords) < 2:
                continue

            start_lon, start_lat = segment_coords[0]
            end_lon, end_lat = segment_coords[-1]

            # Calculate midpoint
            mid_lon = (start_lon + end_lon) / 2
            mid_lat = (start_lat + end_lat) / 2

            # Convert coords to plain Python lists of floats for JSON serialization
            coords_list = [[float(lon), float(lat)] for lon, lat in segment_coords]

            segments.append(
                {
                    "start": [float(start_lon), float(start_lat)],
                    "end": [float(end_lon), float(end_lat)],
                    "midpoint": [float(mid_lon), float(mid_lat)],
                    "coords": coords_list,
                }
            )

    return segments


def haversine_distance_vectorized(
    lon1: float, lat1: float, lon_array: np.array, lat_array: np.array
) -> np.array:
    """
    Calculate haversine distance from one point to multiple points.
    Vectorized for performance.

    Args:
        lon1, lat1: Single point coordinates
        lon_array, lat_array: Arrays of coordinates

    Returns:
        Array of distances in kilometers
    """
    # Convert to radians
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon_array_rad = np.radians(lon_array)
    lat_array_rad = np.radians(lat_array)

    # Haversine formula
    dlon = lon_array_rad - lon1_rad
    dlat = lat_array_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat_array_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c  # Earth radius in km


def check_segment_coverage(
    segment: dict, high_power_chargers_df: pd.DataFrame, max_distance_km: float = 60
) -> bool:
    """
    Check if a road segment has coverage from high-power chargers.

    Args:
        segment: Dictionary with segment information
        high_power_chargers_df: DataFrame of chargers ≥150kW
        max_distance_km: Maximum distance in km to consider coverage

    Returns:
        True if covered, False otherwise
    """
    if high_power_chargers_df.empty:
        return False

    mid_lon, mid_lat = segment["midpoint"]

    # Vectorized distance calculation to all chargers
    distances = haversine_distance_vectorized(
        mid_lon,
        mid_lat,
        high_power_chargers_df["lon"].values,
        high_power_chargers_df["lat"].values,
    )

    # Check if any charger is within range
    return (distances <= max_distance_km).any()


def filter_chargers_near_tent(
    chargers_df: pd.DataFrame, road_segments: list, max_distance_km: float = 3.0
) -> pd.DataFrame:
    """
    Filter chargers to only include those within max_distance_km of the TEN-T network.

    Args:
        chargers_df: DataFrame of chargers to filter
        road_segments: List of road segment dictionaries from process_tent_roads
        max_distance_km: Maximum distance in km to consider a charger near TEN-T

    Returns:
        Filtered DataFrame containing only chargers near TEN-T network
    """
    if chargers_df.empty or not road_segments:
        return chargers_df

    # Extract all points from road segments
    tent_points = []
    for segment in road_segments:
        # Use midpoint of each segment for efficiency
        tent_points.append(segment["midpoint"])

    if not tent_points:
        return chargers_df

    # Convert to numpy arrays for vectorized operations
    tent_lons = np.array([p[0] for p in tent_points])
    tent_lats = np.array([p[1] for p in tent_points])

    # For each charger, find minimum distance to any TEN-T point
    charger_near_tent = []

    for _, charger in chargers_df.iterrows():
        charger_lon = charger["lon"]
        charger_lat = charger["lat"]

        # Calculate distance to all TEN-T points
        distances = haversine_distance_vectorized(
            charger_lon, charger_lat, tent_lons, tent_lats
        )

        # Check if any distance is within threshold
        min_distance = distances.min()
        charger_near_tent.append(min_distance <= max_distance_km)

    # Filter the dataframe
    filtered_df = chargers_df[charger_near_tent].copy()

    return filtered_df


def create_coverage_geojson(
    segments: list, high_power_chargers_df: pd.DataFrame, max_distance_km: float = 60
) -> tuple[dict, dict]:
    """
    Create separate GeoJSON objects for covered (blue) and uncovered (red) segments.

    Returns:
        Tuple of (covered_geojson, uncovered_geojson)
    """
    covered_features = []
    uncovered_features = []

    for segment in segments:
        is_covered = check_segment_coverage(
            segment, high_power_chargers_df, max_distance_km
        )

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": segment[
                    "coords"
                ],  # Already converted to plain Python floats
            },
            "properties": {"covered": bool(is_covered)},
        }

        if is_covered:
            covered_features.append(feature)
        else:
            uncovered_features.append(feature)

    covered_geojson = {"type": "FeatureCollection", "features": covered_features}

    uncovered_geojson = {"type": "FeatureCollection", "features": uncovered_features}

    return covered_geojson, uncovered_geojson


def main():
    """Main function to run the Streamlit application.

    Args:
        None

    Returns:
        None
    """
    st.set_page_config(
        page_title=CONFIG["APP_TITLE"],
        layout="wide",
        page_icon="🔋",
        initial_sidebar_state="expanded",
    )
    st.title(CONFIG["APP_TITLE"])

    st.markdown(
        f"""
        This website shows the progress of the transition to EVs in Germany, built by Janin Jankovski, Marlin Vigelius, Marlon Müller, and Florian Robrecht during the module "Introduction to Data Analytics in Business", tought by Prof. Dr. Lucas Böttcher - datasets last updated on {CONFIG["EV_CHARGER_DATASET_PUBLICATION"].strftime('%Y-%m-%d')}.
        """
    )
    # --- LOAD DATA ---
    ev_station_df = load_and_prepare_data(CONFIG["STATIONS_JSON_PATH"])
    if ev_station_df is not None and not ev_station_df.empty:
        ev_station_df["Inbetriebnahmedatum"] = pd.to_datetime(
            ev_station_df["Inbetriebnahmedatum"], format="%d.%m.%Y"
        )

    ev_registrations_df = pd.read_csv(CONFIG["ELECTRIC_CAR_REGISTRATIONS_PATH"])

    # Initialize filtered dataframe
    ev_station_df_filtered = ev_station_df
    selected_year = None

    # --- CREATE INFO CARDS ---
    st.header("KPIs:")
    # column1: Charging Points, column2: Registered EVs, column3: Registered EVs per public charging point
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            num_charging_points = ev_station_df_filtered["Anzahl Ladepunkte"].sum()
            # EV_CHARGER_DATASET_PUBLICATION - 1 year
            one_year_ago = CONFIG["EV_CHARGER_DATASET_PUBLICATION"].replace(
                year=CONFIG["EV_CHARGER_DATASET_PUBLICATION"].year - 1
            )
            num_charging_points_one_year_ago = ev_station_df[
                ev_station_df["Inbetriebnahmedatum"] <= one_year_ago
            ]["Anzahl Ladepunkte"].sum()

            # Show different label based on whether filtering is applied
            if selected_year and selected_year < max_year:
                label = f"🔋 Public Charging Points in Germany (up to {selected_year})"
            else:
                label = "🔋 Public Charging Points in Germany"

            st.metric(
                label=label,
                value=f"{num_charging_points:,}",
                delta=f"{round(((num_charging_points/num_charging_points_one_year_ago)-1)*100,2)}% YoY",
            )
        except (FileNotFoundError, ValueError):
            st.metric("🔋 Public Charging Points in Germany", "Data not available")
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
                label="🔌 Registered EVs & Plug-In hybrids per public charging point",
                value=round(registered_evs / num_charging_points, 2),
                delta=f"{round(((registered_evs/num_charging_points)/(registered_evs_one_year_ago/num_charging_points_one_year_ago)-1)*100,2)}% YoY",
            )
        except (FileNotFoundError, pd.errors.ParserError):
            st.metric("🔌 Registered EVs & Plug-In hybrids per public charging point", "Data not available")

    # --- BAR CHARTS ---
    st.header("Registrations:")

    # Create three columns for the charts
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Charging Stations")
        # Process charging station data from the JSON file
        if ev_station_df is not None and not ev_station_df.empty:
            # Group by year and count stations
            yearly_stations = (
                ev_station_df.groupby(ev_station_df["Inbetriebnahmedatum"].dt.year)
                .size()
                .reset_index(name="Count")
            )
            yearly_stations = yearly_stations.sort_values("Inbetriebnahmedatum")

            # Prepare data for Streamlit bar chart
            chart_data = yearly_stations.set_index("Inbetriebnahmedatum")["Count"]
            st.bar_chart(chart_data, height=300)
        else:
            st.info("No charging station data available")

    with col2:
        st.subheader("Electric Cars")
        # Process electric car registrations from CSV
        try:
            ev_registrations_df = pd.read_csv(CONFIG["ELECTRIC_CAR_REGISTRATIONS_PATH"])
            electric_data = ev_registrations_df[
                ev_registrations_df["Type"] == "Reine Elektroautos"
            ]

            # Group by year and sum registrations
            yearly_electric = electric_data.groupby("Year")["Count"].sum().reset_index()
            yearly_electric = yearly_electric.sort_values("Year")

            # Prepare data for Streamlit bar chart
            chart_data = yearly_electric.set_index("Year")["Count"]
            st.bar_chart(chart_data, height=300)
        except Exception as e:
            st.info("No electric car registration data available")

    with col3:
        st.subheader("Plug-in Hybrid")
        # Process hybrid car registrations from CSV
        try:
            ev_registrations_df = pd.read_csv(CONFIG["ELECTRIC_CAR_REGISTRATIONS_PATH"])
            hybrid_data = ev_registrations_df[
                ev_registrations_df["Type"] == "Plug-In Hybridautos"
            ]

            # Group by year and sum registrations
            yearly_hybrid = hybrid_data.groupby("Year")["Count"].sum().reset_index()
            yearly_hybrid = yearly_hybrid.sort_values("Year")

            # Prepare data for Streamlit bar chart
            chart_data = yearly_hybrid.set_index("Year")["Count"]
            st.bar_chart(chart_data, height=300)
        except Exception as e:
            st.info("No hybrid car registration data available")

    # --- MONTHLY CHARTS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Monthly New Car Registrations: Electric vs Plug-in Hybrid")

        try:
            # Load and process monthly registration data
            ev_registrations_df = pd.read_csv(CONFIG["ELECTRIC_CAR_REGISTRATIONS_PATH"])

            # Create date column
            ev_registrations_df["Date"] = pd.to_datetime(
                ev_registrations_df["Year"].astype(str)
                + "-"
                + ev_registrations_df["Month"].astype(str)
                + "-01"
            )

            # Filter for electric and hybrid data
            electric_data = ev_registrations_df[
                ev_registrations_df["Type"] == "Reine Elektroautos"
            ]
            hybrid_data = ev_registrations_df[
                ev_registrations_df["Type"] == "Plug-In Hybridautos"
            ]

            # Prepare data for Streamlit line chart
            # Create a combined DataFrame with both types
            monthly_data = (
                pd.DataFrame(
                    {"Date": electric_data["Date"], "Electric": electric_data["Count"]}
                )
                .merge(
                    pd.DataFrame(
                        {
                            "Date": hybrid_data["Date"],
                            "Plug-in Hybrid": hybrid_data["Count"],
                        }
                    ),
                    on="Date",
                    how="outer",
                )
                .fillna(0)
                .sort_values("Date")
            )

            # Set Date as index for Streamlit line chart
            chart_data = monthly_data.set_index("Date")

            # Create the line chart
            st.line_chart(chart_data, height=400)

        except Exception as e:
            st.info("No monthly registration data available")

    with col2:
        st.subheader("📊 Regression Analysis: Charging Stations vs EV Registrations")

        try:
            # Load charging station data
            if ev_station_df is not None and not ev_station_df.empty:
                # Process charging station data by month
                ev_station_df["Date"] = pd.to_datetime(
                    ev_station_df["Inbetriebnahmedatum"], format="%d.%m.%Y"
                )
                ev_station_df["YearMonth"] = ev_station_df["Date"].dt.to_period("M")

                # Group by month and sum charging points
                monthly_charging = (
                    ev_station_df.groupby("YearMonth")["Anzahl Ladepunkte"]
                    .sum()
                    .reset_index()
                )
                monthly_charging["Date"] = monthly_charging[
                    "YearMonth"
                ].dt.to_timestamp()
                monthly_charging.rename(
                    columns={"Anzahl Ladepunkte": "NewChargingPoints"}, inplace=True
                )

                # Load and process registration data
                ev_registrations_df = pd.read_csv(
                    CONFIG["ELECTRIC_CAR_REGISTRATIONS_PATH"]
                )
                ev_registrations_df["Date"] = pd.to_datetime(
                    ev_registrations_df["Year"].astype(str)
                    + "-"
                    + ev_registrations_df["Month"].astype(str)
                    + "-01"
                )

                # Sum all EV registrations (electric + hybrid) by month
                monthly_registrations = (
                    ev_registrations_df.groupby("Date")["Count"].sum().reset_index()
                )
                monthly_registrations.rename(
                    columns={"Count": "TotalNewEVs"}, inplace=True
                )

                # Merge the datasets
                merged_monthly = pd.merge(
                    monthly_charging[["Date", "NewChargingPoints"]],
                    monthly_registrations,
                    on="Date",
                    how="inner",
                )

                # Filter for recent data (from 2015 onwards)
                start_date = pd.to_datetime("2015-01-01")
                merged_monthly = merged_monthly[
                    merged_monthly["Date"] >= start_date
                ].copy()

                if len(merged_monthly) > 1:
                    # Perform linear regression
                    # Swapped: TotalNewEVs is now independent (X), NewChargingPoints is dependent (Y)
                    slope, intercept, r_value, p_value, std_err = linregress(
                        merged_monthly["TotalNewEVs"],
                        merged_monthly["NewChargingPoints"],
                    )

                    # Create regression line data
                    x_line = np.array([0, merged_monthly["TotalNewEVs"].max()])
                    y_line = intercept + slope * x_line

                    # Create data for Altair chart
                    # Prepare scatter plot data
                    scatter_data = merged_monthly[
                        ["TotalNewEVs", "NewChargingPoints"]
                    ].copy()
                    scatter_data.columns = ["x", "y"]

                    # Create regression line data
                    x_min = merged_monthly["TotalNewEVs"].min()
                    x_max = merged_monthly["TotalNewEVs"].max()
                    x_range = np.linspace(x_min, x_max, 50)
                    y_regression = intercept + slope * x_range

                    line_data = pd.DataFrame({"x": x_range, "y": y_regression})

                    # Create base chart
                    base = alt.Chart(scatter_data).encode(
                        x=alt.X("x:Q", title="Total New EV + PHEV Registrations per Month"),
                        y=alt.Y(
                            "y:Q", title="New Charging Points per Month"
                        ),
                    )

                    # Create scatter plot
                    scatter = base.mark_circle(
                        size=60, color="#1f77b4", opacity=0.7
                    ).encode(y="y")

                    # Create regression line
                    line = (
                        alt.Chart(line_data)
                        .mark_line(color="#ff7f0e", strokeWidth=3)
                        .encode(
                            x=alt.X("x:Q", title="Total New EV + PHEV Registrations per Month"),
                            y=alt.Y(
                                "y:Q",
                                title="New Charging Points per Month",
                            ),
                        )
                    )

                    # Combine scatter and line
                    chart = (scatter + line).resolve_scale(x="shared", y="shared")

                    # Display the combined chart
                    st.altair_chart(chart, use_container_width=True)

                    st.caption(
                        f"""
                        **R²:** {r_value**2:.3f} | **Slope:** {slope:.2f} | **P-value:** {p_value:.3f}
                        *(Blue dots = Monthly data points, Orange line = Regression fit)*
                        """
                    )

                else:
                    st.info("Insufficient data for regression analysis")
            else:
                st.info("No charging station data available for regression analysis")

        except Exception as e:
            st.info("Error in regression analysis")

    st.divider()

    # --- INTERACTIVE MAP ---
    st.header("🗺️ Interactive Charging Station Map")
    st.markdown(
        "Explore Germany's EV charging infrastructure with real-time filtering and location search."
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🔍 Map Controls")

        # --- ADDRESS SEARCH ---
        st.subheader("📍 Address Search")
        address_input = st.text_input(
            "Enter a German address:",
            placeholder="Adickesallee 32",
            help="Search for a specific location on the map",
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
        show_low_power = st.checkbox(
            "Stations < 50kW", value=True, help="Show low power charging stations"
        )
        show_medium_power = st.checkbox(
            "Stations ≥ 50kW and < 150kW", value=True, help="Show medium power charging stations"
        )
        show_high_power = st.checkbox(
            "Stations ≥ 150kW", value=True, help="Show high power charging stations"
        )

        # Show power distribution info
        if ev_station_df is not None and not ev_station_df.empty:
            power_col = "Nennleistung Ladeeinrichtung [kW]"

            # Count stations by power level
            low_power_count = len(ev_station_df[ev_station_df[power_col] < 50])
            medium_power_count = len(
                ev_station_df[
                    (ev_station_df[power_col] >= 50) & (ev_station_df[power_col] < 150)
                ]
            )
            high_power_count = len(ev_station_df[ev_station_df[power_col] >= 150])

            st.caption(
                f"""
            **Power Distribution:**
            - < 50kW: {low_power_count:,} stations
            - 50-150kW: {medium_power_count:,} stations  
            - ≥ 150kW: {high_power_count:,} stations
            """
            )

        st.divider()

        st.subheader("🛣️ TEN-T Core Network")
        show_tent = st.checkbox(
            "Show TEN-T core roads", value=True, help="EU core network roads"
        )

        # Checkbox to filter charging stations to only those near TEN-T
        show_only_tent_chargers = st.checkbox(
            "Show only ≥150kW stations near TEN-T (≤3km)",
            value=False,
            help="Filter map to only show high power charging stations within 3km of TEN-T network",
        )

        # Placeholder for coverage stats (will be calculated and displayed later)
        coverage_stats_placeholder = st.empty()

        if show_tent:
            st.caption(
                """
                **Coverage Analysis:**
                - 🔵 Blue: Roads with ≥150kW charging station within 60km
                - 🔴 Red: Roads without coverage
                
                Only charging stations ≥150kW within 3km of TEN-T network are considered.
                Roads are analyzed in 2.5km segments for improved accuracy.
                Coverage updates based on the year filter.
                """
            )

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
                help="Move the slider to see how the charging station network developed over time",
            )

            # Filter data based on selected year
            ev_station_df_filtered = ev_station_df[
                ev_station_df["Inbetriebnahmedatum"].dt.year <= selected_year
            ].copy()

            # Show information about filtered data
            total_stations = len(ev_station_df)
            filtered_stations = len(ev_station_df_filtered)
            if selected_year < max_year:
                st.caption(
                    f"Showing {filtered_stations:,} stations (out of {total_stations:,} total) up to {selected_year}"
                )
            else:
                st.caption(f"Showing all {filtered_stations:,} stations")
        else:
            st.info("No charging station data available")
            selected_year = None
            ev_station_df_filtered = ev_station_df

    # --- APPLY POWER FILTERING ---
    if ev_station_df_filtered is not None and not ev_station_df_filtered.empty:
        power_col = "Nennleistung Ladeeinrichtung [kW]"
        power_mask = pd.Series(
            [False] * len(ev_station_df_filtered), index=ev_station_df_filtered.index
        )

        # Apply power filters
        if show_low_power:
            power_mask |= ev_station_df_filtered[power_col] < 50
        if show_medium_power:
            power_mask |= (ev_station_df_filtered[power_col] >= 50) & (
                ev_station_df_filtered[power_col] < 150
            )
        if show_high_power:
            power_mask |= ev_station_df_filtered[power_col] >= 150

        # Apply the power filter
        ev_station_df_filtered = ev_station_df_filtered[power_mask].copy()

        # Show power filter info in sidebar
        with st.sidebar:
            if not (show_low_power and show_medium_power and show_high_power):
                filtered_power_stations = len(ev_station_df_filtered)
                st.caption(
                    f"⚡ Power filter: {filtered_power_stations:,} stations visible"
                )

    # --- APPLY TEN-T PROXIMITY FILTER ---
    if (
        show_only_tent_chargers
        and ev_station_df_filtered is not None
        and not ev_station_df_filtered.empty
    ):
        # Load TEN-T data and process segments
        ten_t_geojson_for_filter = load_tent_geojson(CONFIG["TEN_T_CORE_GEOJSON_PATH"])
        if ten_t_geojson_for_filter is not None:
            road_segments_for_filter = process_tent_roads(
                ten_t_geojson_for_filter, segment_length_km=2.5
            )

            # First, filter to only high power chargers (≥150kW)
            power_col = "Nennleistung Ladeeinrichtung [kW]"
            high_power_only = ev_station_df_filtered[
                ev_station_df_filtered[power_col] >= 150
            ].copy()

            # Then filter to only those within 3km of TEN-T
            ev_station_df_filtered = filter_chargers_near_tent(
                high_power_only, road_segments_for_filter, max_distance_km=3.0
            )

            # Show info in sidebar
            with st.sidebar:
                st.caption(
                    f"🛣️ TEN-T filter: {len(ev_station_df_filtered):,} stations within 3km of TEN-T"
                )

    # --- CALCULATE TEN-T COVERAGE STATISTICS FOR SIDEBAR ---
    coverage_num_covered = 0
    coverage_total_segments = 0
    coverage_percentage = 0.0

    ten_t_geojson_for_stats = load_tent_geojson(CONFIG["TEN_T_CORE_GEOJSON_PATH"])
    if show_tent and ten_t_geojson_for_stats is not None:
        road_segments_for_stats = process_tent_roads(
            ten_t_geojson_for_stats, segment_length_km=2.5
        )

        # Filter high-power chargers for coverage calculation
        if ev_station_df_filtered is not None and not ev_station_df_filtered.empty:
            power_col = "Nennleistung Ladeeinrichtung [kW]"
            high_power_chargers_stats = ev_station_df_filtered[
                ev_station_df_filtered[power_col] >= 150
            ].copy()

            # Further filter to only chargers within 3km of TEN-T network
            high_power_chargers_stats = filter_chargers_near_tent(
                high_power_chargers_stats, road_segments_for_stats, max_distance_km=3.0
            )
        else:
            high_power_chargers_stats = pd.DataFrame()

        # Calculate coverage
        covered_geojson_stats, uncovered_geojson_stats = create_coverage_geojson(
            road_segments_for_stats, high_power_chargers_stats, max_distance_km=60
        )

        coverage_num_covered = len(covered_geojson_stats["features"])
        coverage_num_uncovered = len(uncovered_geojson_stats["features"])
        coverage_total_segments = coverage_num_covered + coverage_num_uncovered
        coverage_percentage = (
            (coverage_num_covered / coverage_total_segments * 100)
            if coverage_total_segments > 0
            else 0
        )

    # Display coverage in sidebar using the placeholder
    if show_tent and coverage_total_segments > 0:
        with coverage_stats_placeholder.container():
            st.metric(
                label="🛣️ TEN-T Road Coverage",
                value=f"{coverage_percentage:.1f}%",
                help="Percentage of road segments with ≥150kW charging station within 60km",
            )
            st.caption(
                f"**{coverage_num_covered:,}** of **{coverage_total_segments:,}** segments covered"
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

    # Update map view if address was found
    if coords:
        st.session_state.view_state.latitude = coords[0]
        st.session_state.view_state.longitude = coords[1]
        st.session_state.view_state.zoom = 10

    # List to hold all pydeck layers
    layers = []

    # --- CREATE TEN-T CORE NETWORK LAYER WITH COVERAGE ANALYSIS ---
    ten_t_geojson = load_tent_geojson(CONFIG["TEN_T_CORE_GEOJSON_PATH"])

    if show_tent and ten_t_geojson is not None:
        # Pre-process TEN-T roads into segments (cached)
        road_segments = process_tent_roads(ten_t_geojson, segment_length_km=2.5)

        # Filter high-power chargers based on selected year and power
        if ev_station_df_filtered is not None and not ev_station_df_filtered.empty:
            power_col = "Nennleistung Ladeeinrichtung [kW]"
            high_power_chargers = ev_station_df_filtered[
                ev_station_df_filtered[power_col] >= 150
            ].copy()

            # Further filter to only chargers within 3km of TEN-T network
            high_power_chargers = filter_chargers_near_tent(
                high_power_chargers, road_segments, max_distance_km=3.0
            )
        else:
            high_power_chargers = pd.DataFrame()

        # Create coverage GeoJSON (covered in blue, uncovered in red)
        covered_geojson, uncovered_geojson = create_coverage_geojson(
            road_segments, high_power_chargers, max_distance_km=60
        )

        # Create layer for covered segments (blue)
        if covered_geojson["features"]:
            covered_layer = pdk.Layer(
                "GeoJsonLayer",
                data=covered_geojson,
                stroked=True,
                filled=False,
                get_line_color=[30, 100, 210, 180],  # blue
                get_line_width=2,
                line_width_min_pixels=1,
                pickable=False,
                auto_highlight=False,
            )
            layers.insert(0, covered_layer)

        # Create layer for uncovered segments (red)
        if uncovered_geojson["features"]:
            uncovered_layer = pdk.Layer(
                "GeoJsonLayer",
                data=uncovered_geojson,
                stroked=True,
                filled=False,
                get_line_color=[210, 30, 30, 200],  # red
                get_line_width=3,  # Slightly thicker to make it more visible
                line_width_min_pixels=2,
                pickable=False,
                auto_highlight=False,
            )
            layers.insert(0, uncovered_layer)

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
        "html": "<b>{Betreiber}</b><br/>{Nennleistung Ladeeinrichtung [kW]}kW </b><br/> Charging Points: {Anzahl Ladepunkte}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "font-family": "sans-serif",
        },
    }

    # Access Mapbox API key from Streamlit secrets
    mapbox_api_key = os.getenv("MAPBOX_API_KEY")

    st.pydeck_chart(
        pdk.Deck(
            map_style=CONFIG["MAPBOX_STYLE"],
            initial_view_state=st.session_state.view_state,
            layers=layers,
            tooltip=tooltip,
            api_keys={"mapbox": mapbox_api_key},
        ),
        width=1000,
        height=700,
    )


if __name__ == "__main__":
    main()
