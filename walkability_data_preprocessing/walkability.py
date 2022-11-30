import requests
import pandas as pd


place_locations = pd.read_csv("image_location_coordinates.csv")
true_labels = pd.read_csv("true_skill_labels.csv")
walk = pd.read_csv("walkability.csv")
walkability_index = []
place_id = []
safety_score = []
for index, place in place_locations.iterrows():
    # print(index)
    #Get latitude and longitudes
    latitude = str(int(place['lat']))
    longtitude = str(int(place['long']))
    #Get response from API
    response = requests.get("https://geo.fcc.gov/api/census/block/find?latitude=" + latitude + "&longitude=" + longtitude + "&censusYear=2020&showall=false&format=json")
    #Parse json in response
    data = response.json()
    #Print FIPS code
    state = str(data["State"]["FIPS"])
    if state == "None":
        continue
    state = int(state)

    county = str(data['County']['FIPS'])
    county = int(county[3:])

    trace = str(data['Block']['FIPS'])
    trace = int(trace[6:-4])
    block = int(str(trace)[-4:])

    county_walk = walk.loc[(walk['STATEFP'] == state) & (walk['COUNTYFP'] == county)]
    final_walk = county_walk.loc[(county_walk['BLKGRPCE'] == block)]
    if len(final_walk) > 0:
        score = true_labels.loc[true_labels['id'] == place['id']]
        if len(score) > 0:
            print("data")
            safety_score.append(score['rating'])
            place_id.append(place['id'])
            walkability_index.append(final_walk["NatWalkInd"].mean())

final_walkability_scores = pd.DataFrame()
final_walkability_scores['id'] = place_id
final_walkability_scores['walkability_index'] = walkability_index
final_walkability_scores['safety_score'] = safety_score
final_walkability_scores.to_csv("final_walkability_scores.csv")

