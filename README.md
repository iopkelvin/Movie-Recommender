# movie_recommender_hybrid
This project produces a two recommender systems using both recommendation approaches: Collaborative Filtering & Content Based Approach.

Dataset:
IMDB and Movielens: 28 million ratings, 280,000 reviewers, 54,000 movies, 45,000 movies tagged.

0. Recommend.ipynb: Base models (item based and model based apprach in CF recomentations) - Pearson Cor and SVD.
1. Preprocessing: Merging 3 datasets: ratings, movies, tags through movie ID, and it also cleans ratings per person, taking out outliers in ratings per person. 
2. Shortening: Ratings are weighted through percentiles based on total ratings per movie. This was done to scale movies with low number of reviews. Reviews went down from 28m to 14m.
3. Train Embedding: Trains 32 embedding weights using Neural Networks. The weights are updated at each iteration such that the dot product of the two inputs (raterID and MovieID) approach the output (actual - mean rating).
4. Visualize Embedding: Similarity metric used on the embeddings to predict movie recommendations.
5. content_genre_tags: Content Based recommender using genres, tags, and review datasets. Genres and tags are a list of words for each movie. Text preprocessing, bigram, and 1% top uncommon words are filtered out. TFIDF was used to create a BOW for the movies. Identity Matrix and Cosine Similarity were used to find recommendations, and results are sorted by similarity and weighted rating.

There is also a very nice Flask App with a lot of features and attractive design.
# Movie-Recommender
