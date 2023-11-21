import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Remplacez 'your_file.csv' par le chemin de votre fichier CSV
csv_file = 'coordonnees_pays_ville.csv'
df = pd.read_csv(csv_file)
df['Pays'] = df['Pays'].replace('United States of America', 'United States')
# Convertir les colonnes de latitude et longitude en objets géographiques
df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
gdf_points = gpd.GeoDataFrame(df, geometry='geometry')

countries_with_100_points = df['Pays'].value_counts()[df['Pays'].value_counts() >= 100].index.tolist()

for country in countries_with_100_points:
    # Remplacer 'path_to_shapefiles' par le chemin de votre dossier de shapefiles
    shp_file = f'shapefile/{country}.shp'  # Adaptez le format du nom de fichier selon vos shapefiles
    country_shp = gpd.read_file(shp_file)

    # Jointure spatiale pour trouver les points dans chaque région
    points_in_country = gpd.sjoin(gdf_points[gdf_points['Pays'] == country], country_shp, how='inner', op='within')

    # Compter les points par région
    region_counts = points_in_country.groupby('NAME_1').size()
    country_shp['NAME_1'] = region_counts

    #country_shp = country_shp.join(region_counts, on='NAME_1')

    # Créer la carte choroplèthe
    fig, ax = plt.subplots(1, 1)
    plot = country_shp.plot(column=region_counts.name, ax=ax, cmap='Reds', scheme='quantiles', legend=True)
    plt.axis("off")
    #ax.set_xlim([0,180])
    plt.tight_layout()
    plt.savefig(f"chloroplethe {country}.png")


