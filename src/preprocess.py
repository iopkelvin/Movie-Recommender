import random
from functools import lru_cache
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder

#actual path
fileDir = os.path.dirname(os.path.realpath('__file__'))
#correct path
rating_path = os.path.join(fileDir, '../data/ratings.csv')
df = pd.read_csv(rating_path, usecols=['userId', 'movieId', 'rating'])
# Shuffle (reproducibly)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
# Partitioning train/val according to behaviour of keras.Model.fit() when called with
# validation_split kwarg (which is to take validation data from the end as a contiguous
# chunk)
val_split = .05
n_ratings = len(df)
n_train = math.floor(n_ratings * (1-val_split))
itrain = df.index[:n_train]
ival = df.index[n_train:]
# Compactify movie ids.
movie_id_encoder = LabelEncoder()
# XXX: Just fitting globally for simplicity. See movie_helpers.py for more 'principled'
# approach. I don't think there's any realistically useful data leakage here though.
orig_movieIds = df['movieId']
df['orig_movieId'] = orig_movieIds
df['movieId'] = movie_id_encoder.fit_transform(df['movieId'])
# Add centred target variable
df['y'] = df['rating'] - df.loc[itrain, 'rating'].mean()
SCALE = 0
if SCALE:
    # Add version of target variable scale to [0, 1]
    yscaler = sklearn.preprocessing.MinMaxScaler()
    yscaler.fit(df.loc[itrain, 'rating'].values.reshape(-1, 1))
    df['y_unit_scaled'] = yscaler.transform(df['rating'].values.reshape(-1, 1))

def munge_title(title):
    '''
    This function preprocesses the movies dataset
    '''
    i = title.rfind(' (')
    if i != -1:
        title = title[:i]
    for suff_word in ['The', 'A', 'An']:
        suffix = ', {}'.format(suff_word)
        if title.endswith(suffix):
            title = suff_word + ' ' + title[:-len(suffix)]
    return title

def get_year(title):
    l = title.rfind('(') + 1
    try:
        return int(title[l:l+4])
    except ValueError:
        print(title, end='\t')
        return 0

movie_path = os.path.join(fileDir, '../data/movies.csv')
movie_df = pd.read_csv(movie_path)
mdf = movie_df

# XXX: hack
assert mdf.loc[
    mdf.movieId==64997,
    'title'].iloc[0] == 'War of the Worlds (2005)'
mdf.loc[
    mdf.movieId==64997,
    'title'
] = 'War of the Worlds (2005)x'

mdf['orig_movieId'] = mdf['movieId']
n_orig = len(mdf)

# There are some movies listed in movie.csv which have no ratings. Drop them.
whitelist = set(movie_id_encoder.classes_)
mdf = mdf[mdf['movieId'].isin(whitelist)].copy()

# New, compact movie Ids
mdf['movieId'] = movie_id_encoder.transform(mdf['movieId'].values)

mdf = mdf.sort_values(by='movieId').reset_index(drop=True)

# By default use original title field (which includes year of release) as unique key
mdf['key'] = mdf['title']

mdf['year'] = mdf['title'].map(get_year)
mdf['full_title'] = mdf['title']
mdf['title'] = mdf['title'].map(munge_title)

# For movies whose munged title are unique, use it as their key
title_counts = mdf.groupby('title').size()
unique_titles = title_counts.index[title_counts == 1]
unique_ids = mdf.index[mdf.title.isin(unique_titles)]
mdf.loc[unique_ids, 'key'] = mdf.loc[unique_ids, 'title']

mdf['n_ratings'] = df.groupby('movieId').size()
mean_ratings = df.groupby('movieId')['rating'].mean()
mdf['mean_rating'] = mean_ratings

ratings_preprocessed_content_path = os.path.join(fileDir, '../processed_data/ratings_content.csv')
movies_preprocessed_content_path = os.path.join(fileDir, '../processed_data/movies_content.csv')

df.to_csv(ratings_preprocessed_content_path, index=False)
mdf.to_csv(movies_preprocessed_content_path, index=False)
