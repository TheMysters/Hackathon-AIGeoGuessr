import pandas as pd
import requests
import os 
from tqdm import tqdm


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# Remplacez ceci par votre clé API Google Maps
API_KEY = 'AIzaSyAcpe2QI-RoZjfYvtWrmThVAyzaOBJq-tI'

def get_country(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={API_KEY}"
    response = requests.get(url)
    results = response.json()['results']
    for result in results:
        for component in result['address_components']:
            if 'country' in component['types']:
                return component['long_name']
    return None

# Lire le fichier CSV
df = pd.read_csv('coordonnees_resultat.csv')

# Filtrer les lignes sans pays
rows_without_country = df[df['Pays'].isna()]

# Trouver le pays pour chaque point GPS sans pays
for index, row in tqdm(rows_without_country.iterrows(), total=rows_without_country.shape[0]):
    country = get_country(row['Latitude'], row['Longitude'])
    df.at[index, 'Pays'] = country

# Sauvegarder le DataFrame mis à jour
df.to_csv('coordonnees_resultat_google.csv', index=False)
