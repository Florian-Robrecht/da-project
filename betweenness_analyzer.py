import pandas as pd
import osmnx as ox
import networkx as nx

# Set osmnx to cache downloads to speed up subsequent runs
ox.settings.use_cache = True
ox.settings.log_console = True

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

# --- Step 3: Get Road Network ---

# For a quick test, use the small town from your example
place_name = 'Heroldstatt, Germany'

# To run on a larger region (e.g., a state), use:
# place_name = 'Baden-Württemberg, Germany'

# To run on the *entire country* (WILL TAKE A VERY LONG TIME):
# place_name = 'Germany'

print(f"Fetching road network for '{place_name}'...")
# Get the drivable road network
# We use simplify=True to remove non-intersection nodes, making
# the graph smaller and more manageable for centrality calculations.
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)

print(f"Graph for '{place_name}' has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Ensure the graph is suitable for routing by projecting it
# and getting the largest connected component
G_proj = ox.project_graph(G)
# NEW LINE (correct for modern osmnx)
G_strong = ox.get_largest_component(G_proj, strongly=True)