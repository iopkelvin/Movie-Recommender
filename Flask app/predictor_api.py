"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.
This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import numpy as np
import datetime as dt
import pandas as pd
import calendar
import keras


movies_df = pd.read_csv('models/movies_preprocessed.csv')
# with open("models/model_4std_20emb_15po.pkl", "rb") as f:
#     model = pickle.load(f)
model = keras.models.load_model('models/model_4std_20emb_15po.h5')

emb_layer = model.get_layer('movie_embedding')
(w,) = emb_layer.get_weights()

def make_prediction(favofite_movie):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}
    Function makes sure the features are fed to the model in the same order the
    model expects them.
    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    movie = favofite_movie

    threshold = 100
    mainstream_movies = movies_df[movies_df.n_ratings >= threshold].reset_index(drop=True)

    movie_embedding_size = w.shape[1]
    kv = WordEmbeddingsKeyedVectors(movie_embedding_size)
    kv.add(
        mainstream_movies['key'].values,
        w[mainstream_movies.movieId]
    )

    results = kv.most_similar(movie)
    return [result[0] for result in results]

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("These are the recommended movies:")
    results = make_prediction()
    pprint(results)
