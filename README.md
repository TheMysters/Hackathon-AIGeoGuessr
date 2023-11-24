# Hackathon-AIGeoGuessr

Bienvenue dans le projet Hackathon-AIGeoGuessr, une initiative passionnante qui fusionne la technologie de l'intelligence artificielle avec l'expérience de jeu captivante de GeoGuessr.

## Description du Projet

Notre projet vise à repousser les limites du jeu GeoGuessr en intégrant des algorithmes avancés, notamment des réseaux de neurones convolutionnels et des classificateurs. Ces technologies permettent une prédiction précise de la position géographique basée sur des images, ajoutant une nouvelle dimension et un défi supplémentaire au jeu.

## GeoGuessr

GeoGuessr est une expérience de jeu unique qui transporte virtuellement les joueurs n'importe où sur la planète en utilisant des images de Google Maps. Les joueurs explorent des environnements inconnus et doivent deviner leur emplacement en plaçant des marqueurs sur la carte mondiale.

## Modèles

Les modèles développés pour ce projet se basent sur le modèle DensetNet121 et ResNet50.

# Guide d'Utilisation

## Création de l'Environnement Virtuel

1. Ouvrez un terminal dans le répertoire de votre projet.
2. Utilisez la commande suivante pour créer un environnement virtuel (assurez-vous que Python est installé) :

   ```bash
   python -m venv venv

3. Activez l'environnement virtuel: <br>
   * Sur Windows:
   ```bash
   venv\Scripts\activate

  * Sur macOS/Linux:
       ```bash
       source venv/bin/activate

## Installation des Librairies

1. Assurez-vous que votre environnement virtuel est activé.
2. Utilisez la commande suivante pour installer les dépendances à partir du fichier requirements.txt:
   ```bash
   pip install -r requirements.txt

## Lancement de l'Application

1. Une fois les dépendances installées, vous pouvez lancer l'application en utilisant la commande suivante:
   ```bash
   python app.py
2. Accédez à l'application à l'adresse http://localhost:5000 dans votre navigateur web.
