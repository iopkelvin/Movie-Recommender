"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
import pandas as pd


movies = pd.read_csv('models/merged_movies_tags.csv')

with open('models/cosine.pkl', 'rb') as f:
    cosine = pickle.load(f)


# Sorted by Similarity and Rating
titles = movies['key'] # switch to key for no year
indices = pd.Series(movies.index, index=movies['key']) # switch to key for no year

def recommend_by_genre(title):
    idx = indices[title]
    sim_scores = cosine[idx]
    datas = pd.concat([pd.Series(sim_scores), movies['weighted_mean_rating']], axis=1)
    datas.columns = ['similarity', 'weighted_mean_rating']
    datas = datas.sort_values(by=["similarity", 'weighted_mean_rating'], ascending=False)
    index = datas.iloc[1:11].index
    result = titles.iloc[index]
    return list(result.reset_index()['key'])

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("These are the recommended movies:")
    results = recommend_by_genre()
    pprint(results)
