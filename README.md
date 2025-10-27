# 🔋🚗🇩🇪 Public EV Charging Stations in Germany

A comprehensive Streamlit web application that visualizes Germany's electric vehicle charging infrastructure and analyzes its coverage across the TEN-T core road network.

## 🌐 Live Application

**👉 [View the live application here](https://da-project-production.up.railway.app/)**

## 📊 Overview

This interactive dashboard provides insights into Germany's transition to electric vehicles by visualizing:

- **Public EV charging stations** across Germany with real-time filtering
- **TEN-T core road network coverage** analysis showing which segments have adequate charging infrastructure
- **EV registration trends** over time (electric cars vs plug-in hybrids)
- **Statistical analysis** of the relationship between charging infrastructure and EV adoption

## 🎯 Key Features

### Interactive Map
- **Real-time filtering** by charging power levels (< 50kW, 50-150kW, ≥150kW)
- **Address search** to locate specific areas in Germany
- **TEN-T network visualization** with coverage analysis
- **Time-based filtering** to see infrastructure development over years

### Coverage Analysis
- **Blue segments**: Roads with ≥150kW chargers within 60km
- **Red segments**: Roads without adequate coverage
- **2.5km granular analysis** for improved accuracy
- **Strategic charger placement** analysis (only chargers within 3km of TEN-T network considered)

### Data Visualizations
- **KPI metrics**: Total chargers, registered EVs, EVs per charger ratio
- **Registration trends**: Monthly and yearly EV adoption patterns
- **Regression analysis**: Correlation between charging infrastructure and EV registrations

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Mapping**: Pydeck with Mapbox
- **Data Processing**: Pandas, NumPy
- **Geospatial Analysis**: Shapely, GeoJSON
- **Visualization**: Altair
- **Deployment**: Railway

## 📁 Project Structure

```
DA-project/
├── app.py                          # Main Streamlit application
├── data_cleaning.py               # Data preprocessing utilities
├── datasets/                       # Data sources
│   ├── ev_stations.json           # Charging station data
│   ├── ev_registrations_cleaned.csv # EV registration data
│   ├── ten-t.geojson              # TEN-T core network
│   └── ...
├── pages/                          # Additional Streamlit pages
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Florian-Robrecht/da-project.git
   cd DA-project
   ```

2. **Install dependencies**
   ```bash
   # using uv
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "MAPBOX_API_KEY=your_mapbox_api_key" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📈 Data Sources

- **Charging Stations**: Bundesnetzagentur (BNetzA) Ladesäulenregister (August 2025)
- **EV Registrations**: German Federal Motor Transport Authority (KBA)
- **TEN-T Network**: European Commission Trans-European Transport Network

## 🔬 Methodology

### Coverage Analysis
The application analyzes charging infrastructure coverage using:

1. **Segment Creation**: TEN-T roads divided into 2.5km segments for granular analysis
2. **Charger Filtering**: Only high-power chargers (≥150kW) within 3km of TEN-T network
3. **Coverage Detection**: Each segment marked blue if it has a charger within 60km
4. **Strategic Analysis**: Focus on chargers near core road network for long-distance travel

### Performance Optimizations
- **Cached processing** with `@st.cache_data` for fast loading
- **Vectorized distance calculations** using NumPy
- **Efficient geospatial operations** with Shapely

## 👥 Team

Developed by **Janin Jankovski**, **Marlin Vigelius**, **Marlon Müller**, and **Florian Robrecht** during the module "Introduction to Data Analytics in Business" taught by **Prof. Dr. Lucas Böttcher**.

## 🔗 Links

- **Live Application**: [https://da-project-production.up.railway.app/](https://da-project-production.up.railway.app/)

### Data Sources

#### Primary Data Sources
- [BNetzA Ladesäulenregister](https://www.bundesnetzagentur.de/DE/Sachgebiete/ElektrizitaetundGas/Unternehmen_Institutionen/HandelundVertrieb/Ladesaeulenkarte/Ladesaeulenkarte_node.html) - Charging station data
- [KBA Vehicle Registrations](https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/neuzulassungen_node.html) - German vehicle registration data
- [TEN-T Core road network](https://webgate.ec.europa.eu/getis/rest/services/TENTec/tentec_public_services_ext/MapServer/15/query?where=COUNTRY_CODE%3D%27DE%27&outFields=*&returnGeometry=true&f=geojson) - European core transport network

#### Additional Research Sources
- [Electric Vehicles Report - Statista](https://www.statista.com/study/103895/electric-vehicles-report/) - Global EV market analysis and charging infrastructure screenshots
- [Global EV Registrations by Country 2024](https://de.statista.com/statistik/daten/studie/1220664/umfrage/neuzulassungen-von-elektroautos-weltweit-nach-laender/) - International EV adoption comparison
- [World Population Prospects 2024](https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Population) - Population data for per-capita calculations
- [New Electric Car Registrations in Germany](https://de.statista.com/statistik/daten/studie/244000/umfrage/neuzulassungen-von-elektroautos-in-deutschland/) - German EV registration trends
- [New Plug-in Hybrid Registrations in Germany](https://de.statista.com/statistik/daten/studie/1241597/umfrage/neuzulassungen-von-plug-in-hybridfahrzeugen-in-deutschland/) - German PHEV registration data
- [New Charging Points in Germany](https://www.statista.com/statistics/1300745/public-charging-stations-electric-cars-germany/) - German charging infrastructure development
---