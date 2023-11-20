import pandas as pd
import os 

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

df = pd.read_csv("coordonnees_res_nearest_city.csv")

print(df[df["Ville_Proche"].isna()])