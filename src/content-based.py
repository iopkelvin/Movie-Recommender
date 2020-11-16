import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from contextlib import contextmanager
import pickle
import time
import os
import re
import sys
import string
import gensim


def preprocess_movies_tags(create_db = False):
    """
    This function merges movies and tags dataset
    """
    if create_db == True:
        #actual path
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        #correct path
        movies_path = os.path.join(fileDir, '../processed_data/movies_short.csv')
        tags_path = os.path.join(fileDir, '../data/tags.csv')
        tags = pd.read_csv(tags_path)
        movies = pd.read_csv(movies_path)

        # Break up the big genre string into a string array
        movies['genres'] = movies['genres'].str.split('|')
        # Convert genres to string value
        movies['genres'] = movies['genres'].fillna("")

        # Put weight on average rating per movie based on number of ratings
        new = pd.qcut(movies['n_ratings'], [0.1, 0.19,0.31, 0.4, 0.5, 0.6, 0.75,0.85, 0.95, 0.99, 1],
                labels=[0.55, 0.60,0.65, 0.7, 0.75, .8, 0.85, 0.9,0.95,1], duplicates='drop')
        movies['weight_quantile'] = new
        movies['weighted_mean_rating'] = movies['mean_rating'] * movies['weight_quantile'].astype(float)

        # Process tags and merge into movies db
        list_tags = tags.groupby('movieId')['tag'].apply(set).apply(list)
        list_tags = list_tags.reset_index()
        list_tags.columns = ['orig_movieId', 'tag']
        merged_movies = pd.merge(movies, list_tags, how='left', on='orig_movieId')
        merged_movies['genre_tag'] = merged_movies['genres'] + merged_movies['tag'].fillna('').apply(lambda x: list(x))

        alphabetic = lambda x: re.sub('\w*\d\w*', ' ', x.lower())
        text = merged_movies['genre_tag'].astype('str').map(alphabetic)
        merged_movies['genre_tag'] = text

        # Save merged dataset
        merged_path = os.path.join(fileDir, '../processed_data/merged_movies_tags.csv')
        merged_movies.to_csv(merged_path, index=False)
    return

def tfidf_cosine(dataset = False):
    """
    This function does the matrix similarity and returns a pickle
    """
    if dataset == True:
        #actual path
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        #correct path
        merged_path = os.path.join(fileDir, '../processed_data/merged_movies_tags.csv')
        merged_movies = pd.read_csv(merged_path)
        # Create TFIDF vector
        tf = TfidfVectorizer(analyzer='word',min_df=0.008, stop_words='english')
        tfidf_matrix = tf.fit_transform(merged_movies['genre_tag'])
        # Create Cosine Similarity Matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        # Save matrix pickle
        cosine_path = os.path.join(fileDir, '../model/cosine.pkl')
        with open(cosine_path, 'wb') as f:
            pickle.dump(cosine_sim, f)
    return

if __name__ == "__main__":
    preprocess_movies_tags(bool(sys.argv[1]))
    tfidf_cosine(bool(sys.argv[2]))
