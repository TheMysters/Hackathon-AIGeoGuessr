import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import os

def data2country():
    print("Loading data...")
    # Charger les données géographiques des pays
    countries = gpd.read_file('Country_Boundaries/ne_10m_admin_0_countries.shp')
    df = pd.read_csv('coordonnees.csv')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))


    gdf.set_crs(countries.crs, inplace=True)

    #Jointure spatiale
    print("Jointure spatiale...")
    result = gpd.sjoin(gdf, countries, how="left", op='intersects')
    df['Pays'] = result['NAME'] 

    df.to_csv('coordonnees_resultat.csv', index=False,encoding='utf-8')

def data2nearestcity():
    print("import data...")
    cities = gpd.read_file('worldcities.csv',encoding='utf-8')

    df = pd.read_csv('coordonnees_resultat.csv')
    gdf_points = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)])

    # Construction d'un arbre KD pour une recherche rapide des voisins
    print("create tree...")
    tree = cKDTree(cities[['Longitude', 'Latitude']].values)
    print("searching tree...")
    distances, indices = tree.query(gdf_points[['Longitude', 'Latitude']].values, k=1)

    # Ajouter la ville la plus proche et la distance au DataFrame
    print("export...")
    gdf_points['Ville_Proche'] = cities.iloc[indices]['city'].values
    gdf_points.drop(columns=['geometry'],inplace=True)

    # Enregistrer le résultat
    gdf_points.to_csv('coordonnees_res_nearest_city.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    data2nearestcity()