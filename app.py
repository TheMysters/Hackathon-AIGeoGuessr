from flask import Flask, render_template, jsonify, request
from flask_caching import Cache
from fonctions import model_predict

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
    cache.set('image_content', image_content, timeout=300)
    result = model_predict(image_content)
    return jsonify(result=result)

@app.route('/exploration')
def exploration():
    return render_template("exploration.html")

@app.route('/help')
def help():
    return render_template("help.html")

if __name__ == '__main__':
    app.run(debug=True)
