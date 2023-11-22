import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import random
import os
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
        loaded_model = tf.keras.models.load_model("modele/Modele1/")
        return loaded_model

#print(model_predict("0.023286,32.4430804.jpg"))