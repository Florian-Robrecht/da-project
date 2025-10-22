"""
International EV Registrations Page

This page displays international electric vehicle registrations data for 2024,
showing both absolute numbers and per capita statistics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
CONFIG = {
    "WORLDWIDE_DATASET_PATH": "datasets/statistic_id1220664_absatz-von-elektroautos-weltweit-nach-laendern-2024.xlsx",
    "EV_CHARGER_DATASET_PUBLICATION": datetime(2025, 8, 26),
}

COUNTRY_TRANSLATION = {
    "Deutschland": "Germany",
    "Vereinigtes Königreich": "United Kingdom",
    "Frankreich": "France",
    "Kanada": "Canada",
    "Belgien": "Belgium",
    "Niederlande": "Netherlands",
    "Schweden": "Sweden",
    "Südkorea": "South Korea",
    "Norwegen": "Norway",
    "Spanien": "Spain",
    "Italien": "Italy",
    "Dänemark": "Denmark",
    "Schweiz": "Switzerland",
    "Österreich": "Austria",
    "Finnland": "Finland",
    "restliche Welt": "Rest of World",
}


def main():
    """Main function for the International Registrations page."""
    st.set_page_config(
        page_title="International EV Registrations", layout="wide", page_icon="🌍"
    )
    st.title("🌍 International EV Registrations 2024")

    st.markdown(
        f"""
        This page shows the progress of electric vehicle adoption worldwide in 2024.
        """
    )

    # --- LOAD DATA ---
    try:
        worldwide = pd.read_excel(CONFIG["WORLDWIDE_DATASET_PATH"])
        worldwide["Länder"] = (
            worldwide["Länder"].map(COUNTRY_TRANSLATION).fillna(worldwide["Länder"])
        )
    except FileNotFoundError:
        st.error(f"Error: The file '{CONFIG['WORLDWIDE_DATASET_PATH']}' was not found.")
        return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # --- CREATE INFO CARDS ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_registrations = worldwide["2024"].sum()
        st.metric(
            label="Total Global EV Registrations",
            value=f"{int(total_registrations):,}",
            help="Total electric vehicle registrations worldwide in 2024",
        )
    with col2:
        avg_per_country = worldwide["2024"].mean()
        st.metric(
            label="Average per Country",
            value=f"{int(avg_per_country):,}",
            help="Average EV registrations per country",
        )
    with col3:
        top_country = worldwide.loc[worldwide["2024"].idxmax()]
        st.metric(
            label="Leading Country",
            value=top_country["Länder"],
            help="Country with the highest number of EV registrations",
        )

    with col4:
        # Calculate per capita for all countries to find the leader
        worldwide_per_capita = worldwide.copy()
        worldwide_per_capita["registrations_per_1000"] = (
            worldwide_per_capita["2024"] / (worldwide_per_capita["Einwohner"] * 1000000)
        ) * 1000
        top_per_capita_country = worldwide_per_capita.loc[
            worldwide_per_capita["registrations_per_1000"].idxmax()
        ]
        st.metric(
            label="Leader per Capita",
            value=top_per_capita_country["Länder"],
            help="Country with the highest EV registrations per 1,000 inhabitants",
        )

    st.divider()

    # Create toggle for view type
    view_type = st.radio(
        "Select view:",
        ["Absolute Numbers", "Per Capita (per 1,000 inhabitants)"],
        horizontal=True,
        key="international_view",
    )

    # Calculate per capita if needed
    if view_type == "Per Capita (per 1,000 inhabitants)":
        worldwide["registrations_per_1000"] = (
            worldwide["2024"] / (worldwide["Einwohner"] * 1000000)
        ) * 1000
        data_column = "registrations_per_1000"
        y_label = "Registrations per 1,000 inhabitants"
        title_suffix = "per capita"
        # Sort by per capita for better visualization
        worldwide_sorted = worldwide.sort_values(
            "registrations_per_1000", ascending=False
        )
    else:
        data_column = "2024"
        y_label = "Total Registrations"
        title_suffix = "absolute numbers"
        # Sort by absolute numbers for better visualization
        worldwide_sorted = worldwide.sort_values("2024", ascending=False)

    # Create the bar chart using Streamlit
    st.subheader(f"Electric car registrations by country 2024 ({title_suffix})")
    
    # Prepare data for Streamlit bar chart - create a DataFrame with proper ordering
    chart_df = worldwide_sorted[["Länder", data_column]].copy()
    chart_df = chart_df.set_index("Länder")
    
    # Use Streamlit's native bar chart
    st.bar_chart(chart_df, height=500)


if __name__ == "__main__":
    main()
