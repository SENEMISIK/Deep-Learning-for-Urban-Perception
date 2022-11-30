import numpy as np
import pandas as pd 
import requests
from trueskill import Rating
import trueskill


if __name__ == '__main__':
    votes = pd.read_csv("./raw_data/votes_clean.csv")
    votes = votes[votes["study_question"] == "safer"]
    left = votes[["left"]]
    left = left.rename({"left": "id"}, axis='columns')
    right = votes[["right"]]
    right = right.rename({"right": "id"}, axis='columns')
    labels = left.append(right, ignore_index=True)
    labels = labels.drop_duplicates()
    
    rankings = {}
    for index, id in labels.iterrows():
        if id['id'] not in rankings:
            rankings[id['id']] = Rating()

    for index, vote in votes.iterrows():
        print(index)
        left = vote['left']
        right = vote['right']
        choice = vote['choice']
        if left in rankings and right in rankings:
            if choice == "left":
                prev = rankings[left]
                rankings[left], rankings[right] = trueskill.rate_1vs1(rankings[left], rankings[right])
                assert rankings[left] > prev
            elif choice == "right":
                prev = rankings[right]
                rankings[right], rankings[left] = trueskill.rate_1vs1(rankings[right], rankings[left])
                assert rankings[right] > prev
            else:
                rankings[right], rankings[left] = trueskill.rate_1vs1(rankings[right], rankings[left],  drawn=True)

    votes = votes[(votes["place_name_right"] == "New York") | (votes["place_name_right"] == "Boston")]
    votes = votes[(votes["place_name_left"] == "Boston") | (votes["place_name_left"] == "New York")]
    left = votes[["left"]]
    left = left.rename({"left": "id"}, axis='columns')
    right = votes[["right"]]
    right = right.rename({"right": "id"}, axis='columns')
    final_labels = left.append(right, ignore_index=True)
    final_labels = labels.drop_duplicates()

    column = np.zeros((len(final_labels)))
    i = 0
    for index, location in final_labels.iterrows():
        column[i] = rankings[location['id']].mu
        i += 1
        print(i)
    
    column -= min(column)
    column *= 10/max(column)

    final_labels["rating"] = column
    final_labels.to_csv("final_labels.csv")
    
