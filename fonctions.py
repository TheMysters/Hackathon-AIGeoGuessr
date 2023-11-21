import random

def model_predict(file_path: str):
    min_lat, max_lat = -90, 90
    min_lng, max_lng = -180, 180

    random_lat = random.uniform(min_lat, max_lat)
    random_lng = random.uniform(min_lng, max_lng)

    result = (random_lat, random_lng)
    return result