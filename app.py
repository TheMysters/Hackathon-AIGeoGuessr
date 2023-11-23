from flask import Flask, render_template, jsonify, request
from flask_caching import Cache
from fonctions import model_predict_DenseNet121
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input
from io import BytesIO
import numpy as np

app = Flask(__name__)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
def main():
    return render_template("home.html")

@app.route('/game1')
def game1():
    return render_template("game1.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    if 'image' not in request.files:
        return jsonify(error='No file provided'), 400

    image_file = request.files['image']

    # Check if the file has a valid filename
    if image_file.filename == '':
        return jsonify(error='Empty filename'), 400

    image_content = image_file.read()
    image = Image.open(BytesIO(image_content))
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    cache.set('image_array', image_array, timeout=300)
    result = model_predict_DenseNet121(image_array)
    result = tuple(float(value) for value in result)
    return jsonify(result=result)

@app.route('/exploration')
def exploration():
    return render_template("exploration.html")

@app.route('/stats_densetnet121')
def stats_densetnet121():
    return render_template("stats_densetnet121.html")

@app.route('/stats_resnet50')
def stats_resnet50():
    return render_template("stats_resnet50.html")

@app.route('/help')
def help():
    return render_template("help.html")

if __name__ == '__main__':
    app.run(debug=True)
