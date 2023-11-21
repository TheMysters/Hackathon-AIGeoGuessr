from flask import Flask, render_template
from folium import Map, Marker, Popup
import pandas as pd

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("home.html")

@app.route('/exploration')
def exploration():
    return render_template("exploration.html")

if __name__ == '__main__':
    app.run(debug=True)
