import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import os
import random, csv
import os, re
from geopy.distance import geodesic
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

MODEL_DIR = "modele/ModeleRegion"
CLASSIFIER_MODEL = "modele/ModeleRegion/classi_test.h5"

ORDER = ['Europe East', 'Europe West','India','Japan',  'North America', 'South Africa','South America', 'South Asia']


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


def load_all_model():
    models = {}
    # Charger le modèle de classification des régions avec des objets personnalisés
    print("loading classifier...")
    models["classifier"] = tf.keras.models.load_model(CLASSIFIER_MODEL)
    with tf.keras.utils.custom_object_scope({'haversine_loss': haversine_loss}):
        # Charger les modèles régionaux
        for region in ['Europe West', 'Europe East', 'North America', 'South America', 'South Africa', 'India', 'Japan', 'South Asia']:
            print(f"loading {region}...")
            model_path = os.path.join(MODEL_DIR, region, 'model.h5')
            models[region] = tf.keras.models.load_model(model_path)

    return models

def predict(models, img_path):
    # Préparer l'image
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction de la région
    region_prediction = models["classifier"].predict(img_array)
    indice = np.argmax(region_prediction[0])
    region = ORDER[indice]

    # Prédiction avec le modèle régional
    regional_prediction = models[region].predict(img_array)

    return region, regional_prediction


def evaluate(test_dir, models):
    total_loss = 0
    num_images = 0

    for file in os.listdir(test_dir):
        if file.endswith(".jpg"):
            # Extraire les vraies valeurs de latitude et longitude du nom du fichier
            lat_true, lon_true = map(float, file[:-4].split(','))

            # Chemin complet de l'image
            img_path = os.path.join(test_dir, file)

            # Prédire la latitude et longitude
            _, regional_prediction = predict(models, img_path)
            lat_pred, lon_pred = regional_prediction[0][0], regional_prediction[0][1]

            # Convertir les vraies valeurs en un format compatible avec la fonction haversine_loss
            y_true = np.array([[lat_true, lon_true]])
            y_pred = np.array([[lat_pred, lon_pred]])

            # Calculer la perte Haversine
            loss = haversine_loss(y_true, y_pred).numpy()
            total_loss += loss
            num_images += 1

    # Calculer la perte moyenne
    mean_loss = total_loss / num_images if num_images > 0 else 0

    return mean_loss


def evaluate_model_regional(models, folder_path):
    with open("model_regional.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['LatitudeOrigine', 'LongitudeOrigine', 'LatitudeModel', 'LongitudeModel', 'DistanceModel', 'LatitudeRandom', 'LongitudeRandom', 'DistanceRandom'])
        pattern = re.compile(r'([-+]?\d*\.\d+|\d+),([-+]?\d*\.\d+|\d+)\.jpg')
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                match = pattern.match(filename)
                if match:
                    latitude_origin = float(match.group(1))
                    longitude_origin = float(match.group(2))

                    _, regional_prediction = predict(models, os.path.join(folder_path, filename))
                    latitude_model, longitude_model = regional_prediction[0][0], regional_prediction[0][1]

                    distance_model_km = geodesic((latitude_origin, longitude_origin), (latitude_model, longitude_model)).kilometers

                    latitude_random = random.uniform(-90, 90)
                    longitude_random = random.uniform(-180, 180)

                    distance_random_km = geodesic((latitude_origin, longitude_origin), (latitude_random, longitude_random)).kilometers
                    csv_writer.writerow([latitude_origin, longitude_origin, latitude_model, longitude_model, distance_model_km, latitude_random, longitude_random, distance_random_km])

def add_region_classifier(csv_path, folder_path, models):
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        # Extraire les informations de chaque ligne
        latitude_origine = row['LatitudeOrigine']
        longitude_origine = row['LongitudeOrigine']
        file_name = f"{latitude_origine},{longitude_origine}.jpg"
        file_path = os.path.join(folder_path, file_name)

        region, _ = predict(models, file_path)

        df.at[index, 'RegionClassifier'] = region

    df.to_csv('votre_nouveau_fichier.csv', index=False)

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes),
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    plt.xticks(rotation=33, ha='right', rotation_mode='anchor', fontsize=text_size)
    plt.yticks(rotation=33, ha='right', rotation_mode='anchor', fontsize=text_size)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
    plt.show()

def read_prediction_classifier_csv(csv_filename):
    y_true = []
    y_preds = []

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            y_true.append(row['RegionOrigine'])
            y_preds.append(row['RegionClassifier'])

    return y_true, y_preds

def make_confusion_matrix_classifier():
    y_true, y_preds = read_prediction_classifier_csv("historique_modelRegion.csv")

    class_names = ['Europe East', 'Europe West', 'India', 'Japan', 'North America',
                'South Africa', 'South America', 'South Asia']
    
    make_confusion_matrix(y_true=y_true,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(10, 10),
                        text_size=8)
    
def read_prediction_pipeline_csv(csv_filename):
    y_true = []
    y_preds = []

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            y_true.append(row['RegionOrigine'])
            y_preds.append(row['RegionModel'])

    return y_true, y_preds

def make_confusion_matrix_pipeline():
    y_true, y_preds = read_prediction_pipeline_csv("historique_modelRegion.csv")

    class_names = ['Europe East', 'Europe West', 'India', 'Japan', 'North America', 'Ocean',
                'South Africa', 'South America', 'South Asia']

    make_confusion_matrix(y_true=y_true,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(10, 10),
                        text_size=8)

make_confusion_matrix_pipeline()