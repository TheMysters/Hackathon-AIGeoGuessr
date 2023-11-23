import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, csv
import os, re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def model_predict(image_content):
    loaded_model = load_model()
    result = loaded_model.predict(image_content) 
    result_tuple = tuple(result[0])
    return (result_tuple[0], result_tuple[1])

def haversine_loss(y_true, y_pred):
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]

    # Conversion de degrés en radians
    lat_true, lon_true, lat_pred, lon_pred = [x * (tf.constant(np.pi) / 180) for x in [lat_true, lon_true, lat_pred, lon_pred]]

    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    a = tf.math.sin(dlat / 2) ** 2 + tf.math.cos(lat_true) * tf.math.cos(lat_pred) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    R = 6371  # Rayon de la Terre en kilomètres
    return R * c

def load_model():
    with tf.keras.utils.custom_object_scope({'haversine_loss': haversine_loss}):
        loaded_model = tf.keras.models.load_model("modele/DenseNet_omg/model_checkpoint_DenseNet121_epoch_18.h5")
        return loaded_model

def evaluate_model(folder_path):
    loaded_model = load_model()
    with open("historique.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['LatitudeOrigine', 'LongitudeOrigine', 'LatitudeModel', 'LongitudeModel', 'DistanceModel', 'LatitudeRandom', 'LongitudeRandom', 'DistanceRandom'])
        pattern = re.compile(r'([-+]?\d*\.\d+|\d+),([-+]?\d*\.\d+|\d+)\.jpg')
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                match = pattern.match(filename)
                if match:
                    latitude_origin = float(match.group(1))
                    longitude_origin = float(match.group(2))

                    input_img = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
                    input_arr = img_to_array(input_img) / 255.0
                    input_arr = np.expand_dims(input_arr, axis=0)
                    result = loaded_model.predict(input_arr) 
                    result_tuple = tuple(result[0])
                    latitude_model, longitude_model = result_tuple[0], result_tuple[1]

                    distance_model_km = geodesic((latitude_origin, longitude_origin), (latitude_model, longitude_model)).kilometers

                    latitude_random = random.uniform(-90, 90)
                    longitude_random = random.uniform(-180, 180)

                    distance_random_km = geodesic((latitude_origin, longitude_origin), (latitude_random, longitude_random)).kilometers
                    csv_writer.writerow([latitude_origin, longitude_origin, latitude_model, longitude_model, distance_model_km, latitude_random, longitude_random, distance_random_km])

def visualize_evaluation():
    df = pd.read_csv('historique.csv')
    bin_width = 500
    bins = np.arange(0, df['DistanceModel'].max() + bin_width, bin_width)

    plt.hist(df['DistanceModel'], bins=bins, alpha=0.5, label='DistanceModel')
    plt.hist(df['DistanceRandom'], bins=bins, alpha=0.5, label='DistanceRandom')

    mean_model = df['DistanceModel'].mean()
    mean_random = df['DistanceRandom'].mean()

    plt.axvline(x=mean_model, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne du Modèle: {mean_model:.2f}')
    plt.axvline(x=mean_random, color='green', linestyle='dashed', linewidth=2, label=f'Moyenne Aléatoire: {mean_random:.2f}')

    plt.xlabel('Distance (km)')
    plt.ylabel('Nombre de Samples')
    plt.title('Histogramme de DistanceModel et DistanceRandom avec Moyennes')   

    plt.legend()
    plt.show()

def add_country_from_coordinates_origin(csv_file): #Utiliser Coord2Country.py de Léo dans Coord_Data
    df = pd.read_csv(csv_file)
    geolocator = Nominatim(user_agent="get_country_app")
    df['Pays'] = ""

    for index, row in df.iterrows():
        latitude = row['LatitudeOrigine']
        longitude = row['LongitudeOrigine']
        location = geolocator.reverse((latitude, longitude), language='en')
        country = location.raw.get('address', {}).get('country', 'Pays inconnu')
        df.at[index, 'Pays'] = country
        print(country)

    df.to_csv('tt.csv', index=False)

def add_country_from_coordinates_model(csv_file): #Utiliser Coord2Country.py de Léo dans Coord_Data
    df = pd.read_csv(csv_file)
    geolocator = Nominatim(user_agent="get_country_app")
    df['PaysModel'] = ""

    for index, row in df.iterrows():
        latitude = row['LatitudeModel']
        longitude = row['LongitudeModel']
        location = geolocator.reverse((latitude, longitude), language='en')
        if location is not None:
            country = location.raw.get('address', {}).get('country', 'Ocean')
        else:
            country = 'Ocean'
        df.at[index, 'PaysModel'] = country
        print(country)

    df.to_csv('tt.csv', index=False)

def visualize_evaluation_country(pays_name:str):
    df = pd.read_csv('historique_pays_all.csv')
    df_us = df[df['PaysOrigine'] == pays_name]
    if df_us.empty:
        print(f"Aucune donnée disponible pour les {pays_name}.")
        return

    bin_width = 500
    bins = np.arange(0, max(df_us['DistanceModel'].max(), df_us['DistanceRandom'].max()) + bin_width, bin_width)

    plt.hist(df_us['DistanceModel'], bins=bins, alpha=0.5, label='DistanceModel')
    plt.hist(df_us['DistanceRandom'], bins=bins, alpha=0.5, label='DistanceRandom')

    mean_model = df_us['DistanceModel'].mean()
    mean_random = df_us['DistanceRandom'].mean()

    plt.axvline(x=mean_model, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne du Modèle: {mean_model:.2f}')
    plt.axvline(x=mean_random, color='green', linestyle='dashed', linewidth=2, label=f'Moyenne Aléatoire: {mean_random:.2f}')

    plt.xlabel('Distance (km)')
    plt.ylabel('Nombre de Samples')
    plt.title(f'Histogramme de DistanceModel et DistanceRandom pour les {pays_name} avec Moyennes')

    plt.legend()
    plt.show()

def visualize_confused_countries(csv_file, country:str):
    df = pd.read_csv(csv_file)

    # Filtrer le DataFrame pour les cas où le pays d'origine est les États-Unis et le pays modèle n'est pas les États-Unis
    confused_cases = df[(df['PaysOrigine'] == country) & (df['PaysModel'] != 'Ocean')]

    plt.figure(figsize=(10, 6))
    confused_cases['PaysModel'].value_counts().plot(kind='bar', color='skyblue')

    plt.xlabel(f'Pays Modèle lorsqu\'Origine est {country}')
    plt.ylabel('Nombre de Cas')
    plt.title(f'Affichage des pays par le modele là où le Pays d\'Origine est {country}')

    plt.show()

#visualize_evaluation_country('India')
#visualize_confused_countries('historique_pays_all.csv', 'India')

def generate_map(csv_filename, country_name):
    # Charger le fichier CSV
    df = pd.read_csv(csv_filename)

    # Filtrer le DataFrame en fonction du pays d'origine
    df_filtered = df[df['PaysOrigine'] == country_name]

    # Vérifier si des données sont disponibles pour le pays spécifié
    if df_filtered.empty:
        print(f"Aucune donnée disponible pour le pays d'origine '{country_name}'.")
        return

    # Créer une carte centrée sur la première paire de coordonnées
    m = folium.Map(location=[df_filtered['LatitudeOrigine'].iloc[0], df_filtered['LongitudeOrigine'].iloc[0]], zoom_start=2)

    # Ajouter des marqueurs bleus pour les coordonnées d'origine
    for index, row in df_filtered.iterrows():
        folium.Marker(location=[row['LatitudeOrigine'], row['LongitudeOrigine']],
                      popup=f"Pays d'Origine: {row['PaysOrigine']}",
                      icon=folium.Icon(color='blue')).add_to(m)

    # Ajouter des icônes spéciales (feuille rouge) pour les coordonnées du modèle
    for index, row in df_filtered.iterrows():
        folium.Marker(location=[row['LatitudeModel'], row['LongitudeModel']],
                      popup=f"Pays prédit par le Modèle: {row['PaysModel']}",
                      icon=folium.Icon(color='red')).add_to(m)

    # Sauvegarder la carte dans un fichier HTML
    m.save("map.html")

#generate_map('historique_pays_all.csv', 'United States')
