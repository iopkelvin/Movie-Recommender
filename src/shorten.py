import pandas as pd
import numpy as np
import os
import sys

# Actual path
fileDir = os.path.dirname(os.path.realpath('__file__'))
# Correct path
movies_path = os.path.join(fileDir, '../processed_data/movies_content.csv')
ratings_path = os.path.join(fileDir, '../processed_data/ratings_content.csv')
tags_path = os.path.join(fileDir, '../data/tags.csv')

def shorten_rating_db(counts, std):
    '''
    This function takes in the processed rating and movies data, and it returns both datasets filtered
    on the amount of ratings that they have.
    parameters:
    counts: Movies that have more than "counts" amount of ratings
    std: Filters out movies that are 'std' times number of standard deviations away from the average number of reviews per reviewer.
    '''
    # Read files
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    ## Reducing movies and ratings that have more than 5 ratings per movie
    subset_movies_by_count = (ratings.groupby('movieId').count() > counts)['userId']
    # Subsetting movies which have been rarely rated
    movie_id = subset_movies_by_count.index[subset_movies_by_count]
    movies = movies[movies.movieId.isin(movie_id)]
    ratings = ratings[ratings.movieId.isin(movie_id)]

    ## Filtering person-ratings that are x STD away from the median rating per person. (e.g. Some people have rated too many times)
    # Median amount of ratings per person
    median = ratings.groupby('userId')['rating'].count().median()
    iqr = ratings.groupby('userId')['rating'].count().quantile(.75) - ratings.groupby('userId')['rating'].count().quantile(.25)
    # Find outliers given number of standard deviations
    outliers = median + std * iqr ###only super extreme users
    subset_users_by_count = (ratings.groupby('userId')['rating'].count() < outliers)
    user_id = subset_users_by_count.index[subset_users_by_count]
    ratings = ratings[ratings.userId.isin(user_id)]

    #Subsetting reviews whos reviewers have more than 2 reviews
    subset_users_by_count_min = (ratings.groupby('userId')['rating'].count() > 2)
    user_id_min = subset_users_by_count_min.index[subset_users_by_count_min]
    ratings = ratings[ratings.userId.isin(user_id_min)]

    # Save Ratings csv
    rating_preprocessed_path = os.path.join(fileDir, '../processed_data/ratings_std.csv')
    ratings.to_csv(path_or_buf=rating_preprocessed_path, index=False)
    # Save movies csv
    movies_preprocessed_path = os.path.join(fileDir, '../processed_data/movies_short.csv')
    movies.to_csv(path_or_buf=movies_preprocessed_path, index=False)
    return

if __name__ == '__main__':
    shorten_rating_db(int(sys.argv[1]), int(sys.argv[2]))
