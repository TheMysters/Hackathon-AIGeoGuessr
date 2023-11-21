import os

def extract_country_names(directory):
    countries = []
    for file in os.listdir(directory):
        if file.endswith('.png'):
            # Suppression du préfixe et de l'extension
            country_name = file.replace('chloroplethe ', '').rsplit('.', 1)[0]
            countries.append(country_name)
    return countries

country_list = extract_country_names('static/VisualisationDeDonnee/Entree/cloroplethe')
print(country_list)
['Argentina', 'Australia', 'Brazil', 'Canada', 'Chile', 'Colombia', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Norway', 'Philippines', 'Poland', 'Romania', 'Russia', 'South', 'Spain', 'Sweden', 'Thailand', 'Turkey', 'United', 'United']