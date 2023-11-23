import pandas as pd
import os
import pandas as pd
from shutil import copyfile, move

def get28MostCountries():
    input_csv = 'coordonnees_pays_ville.csv'
    df = pd.read_csv(input_csv)

    # Compter le nombre de points par pays
    points_count = df['Pays'].value_counts()

    # Filtrer les pays avec au moins 50 points
    filtered_countries = points_count[points_count >= 100].index

    # Créer un nouveau DataFrame avec les données filtrées
    output_df = df[df['Pays'].isin(filtered_countries)]

    # Sauvegarder le nouveau CSV
    output_csv = 'nouveau_fichier.csv'
    output_df.to_csv(output_csv, index=False)

def split_folder(folder_path: str):
    # Charger le fichier CSV qui contient les informations sur les pays
    csv_path = 'coord_region.csv'
    df = pd.read_csv(csv_path)

    os.makedirs(folder_path, exist_ok=True)

    # Itérer à travers les lignes du DataFrame
    for index, row in df.iterrows():
        country_folder = "images_region/training/"
        lat, lon, region = row['Latitude'], row['Longitude'], row['Region']
        image_filename = f"{lat},{lon}.jpg"
        image_path = os.path.join(country_folder, image_filename)

        # Vérifier si l'image existe avant de la déplacer
        if os.path.exists(image_path):
            country_folder = os.path.join(folder_path, region)
            os.makedirs(country_folder, exist_ok=True)

            # Copier l'image dans le dossier du pays
            destination_path = os.path.join(country_folder, image_filename)
            copyfile(image_path, destination_path)
            print(f"Image {image_filename} déplacée vers {country_folder}")
        else:
            print(f"Image {image_filename} non trouvée dans {country_folder}")

    print("Opération terminée.")

#split_folder('images_region/')

def add_region(csv_path):
    df = pd.read_csv(csv_path)

    regions = {
        'Europe West': ['France', 'Spain', 'Italy', 'Turkey', 'Germany', 'Poland', 'United Kingdom', 'Netherlands'],
        'Europe East': ['Sweden', 'Norway', 'Finland', 'Romania', 'Russia'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'South America': ['Brazil', 'Chile', 'Argentina', 'Colombia'],
        'South Africa': ['South Africa'],
        'India': ['India'],
        'Japan': ['Japan'],
        'South Asia': ['Indonesia', 'Philippines', 'Thailand']
    }
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return 'Ocean'
 
    # Ajout des colonnes RegionOrigine et RegionModel
    df['RegionOrigine'] = df['PaysOrigine'].apply(get_region)
    df['RegionModel'] = df['PaysModel'].apply(get_region)

    # Sauvegarde du DataFrame modifié dans un nouveau fichier CSV
    df.to_csv('nouveau_fichier.csv', index=False)

import shutil
from sklearn.model_selection import train_test_split

def logique(region: str):
    df = pd.read_csv('coord_region.csv')

    # Filtrer les données pour ne prendre que celles de la région South Africa
    south_africa_data = df[df['Region'] == region]

    # Diviser les données en training (80%), test (10%), et validation (10%)
    train_data, test_and_val_data = train_test_split(south_africa_data, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(test_and_val_data, test_size=0.5, random_state=42)

    # Créer les dossiers de destination s'ils n'existent pas déjà
    for folder in ['training', 'test', 'validation']:
        os.makedirs(os.path.join('images_region', folder), exist_ok=True)

    # Copier les fichiers d'images dans les dossiers appropriés
    for index, row in train_data.iterrows():
        image_filename = f"{row['Latitude']},{row['Longitude']}.jpg"
        source_path = os.path.join('images_region', image_filename)
        dest_path = os.path.join('images_region', 'training', region ,image_filename)
        shutil.copy(source_path, dest_path)

    for index, row in test_data.iterrows():
        image_filename = f"{row['Latitude']},{row['Longitude']}.jpg"
        source_path = os.path.join('images_region', image_filename)
        dest_path = os.path.join('images_region', 'test', region ,image_filename)
        shutil.copy(source_path, dest_path)

    for index, row in val_data.iterrows():
        image_filename = f"{row['Latitude']},{row['Longitude']}.jpg"
        source_path = os.path.join('images_region', image_filename)
        dest_path = os.path.join('images_region', 'validation', region ,image_filename)
        shutil.copy(source_path, dest_path)

