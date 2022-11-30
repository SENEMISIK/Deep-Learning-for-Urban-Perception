import numpy as np
import pandas as pd 
import requests


if __name__ == '__main__':
    df = pd.read_csv("votes_clean.csv")
    df = df[df["study_question"] == "safer"]
    df = df[(df["place_name_right"] == "New York") | (df["place_name_right"] == "Boston")]
    df = df[(df["place_name_left"] == "Boston") | (df["place_name_left"] == "New York")]
    df.to_csv("survey.csv")

    left = df[["left", "long_left", "lat_left"]]
    left = left.rename({"left": "id", "long_left":"long", "lat_left":"lat"}, axis='columns')
    right = df[["right", "long_right", "lat_right"]]
    right = right.rename({"right": "id", "long_right":"long", "lat_right":"lat"}, axis='columns')
    place_locations = left.append(right, ignore_index=True)
    place_locations = place_locations.drop_duplicates()
    place_locations.to_csv("place_locations.csv")
    i = 0
    for index, place in place_locations.iterrows():
        print(i)
        i += 1
        id = place['id']
        long = place['long']
        lat = place['lat']
        try:
            receive = requests.get(f"https://maps.googleapis.com/maps/api/streetview?size=400x300&location={long},{lat}&key=AIzaSyBqxBPISNtr5lQp-e0Bhd1NvfHas5NMzac")
            with open(f"images/{id}.png",'wb') as f:
                f.write(receive.content)
        except:
            print("Failed for id: {id}, long: {long}, lat: {lat}")
