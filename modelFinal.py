import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

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
