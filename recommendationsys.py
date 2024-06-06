import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load movie ratings dataset
movie_ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")

# Display first few rows of the movie ratings dataset
print("First few rows of the movie ratings dataset:")
print(movie_ratings.head())

# Load movie metadata dataset
movie_metadata = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

# Display first few rows of the movie metadata dataset
print("\nFirst few rows of the movie metadata dataset:")
print(movie_metadata.head())

# Calculate basic statistics of the dataset
num_ratings = len(movie_ratings)
num_movies = len(movie_ratings['movieId'].unique())
num_users = len(movie_ratings['userId'].unique())
avg_ratings_per_user = round(num_ratings / num_users, 2)
avg_ratings_per_movie = round(num_ratings / num_movies, 2)

print(f"\nNumber of ratings: {num_ratings}")
print(f"Number of unique movies: {num_movies}")
print(f"Number of unique users: {num_users}")
print(f"Average ratings per user: {avg_ratings_per_user}")
print(f"Average ratings per movie: {avg_ratings_per_movie}")

# Calculate user rating frequency
user_rating_freq = movie_ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_rating_freq.columns = ['userId', 'num_ratings']

print("\nUser rating frequency:")
print(user_rating_freq.head())

# Find lowest and highest rated movies
mean_movie_ratings = movie_ratings.groupby('movieId')[['rating']].mean()
lowest_rated_movie_id = mean_movie_ratings['rating'].idxmin()
highest_rated_movie_id = mean_movie_ratings['rating'].idxmax()

lowest_rated_movie_title = movie_metadata.loc[movie_metadata['movieId'] == lowest_rated_movie_id, 'title'].values[0]
highest_rated_movie_title = movie_metadata.loc[movie_metadata['movieId'] == highest_rated_movie_id, 'title'].values[0]

num_users_highest_rated = len(movie_ratings[movie_ratings['movieId'] == highest_rated_movie_id])
num_users_lowest_rated = len(movie_ratings[movie_ratings['movieId'] == lowest_rated_movie_id])

print(f"\nLowest rated movie: {lowest_rated_movie_title}")
print(f"Highest rated movie: {highest_rated_movie_title}")
print(f"Number of users who rated the highest rated movie: {num_users_highest_rated}")
print(f"Number of users who rated the lowest rated movie: {num_users_lowest_rated}")

# Create user-item matrix using sparse matrix
def create_user_item_matrix(df):
    num_users = len(df['userId'].unique())
    num_movies = len(df['movieId'].unique())

    user_id_to_index = dict(zip(np.unique(df['userId']), range(num_users)))
    movie_id_to_index = dict(zip(np.unique(df['movieId']), range(num_movies)))

    user_index_to_id = dict(zip(range(num_users), np.unique(df['userId'])))
    movie_index_to_id = dict(zip(range(num_movies), np.unique(df['movieId'])))

    user_indices = [user_id_to_index[i] for i in df['userId']]
    movie_indices = [movie_id_to_index[i] for i in df['movieId']]

    user_item_matrix = csr_matrix((df["rating"], (movie_indices, user_indices)), shape=(num_movies, num_users))

    return user_item_matrix, user_id_to_index, movie_id_to_index, user_index_to_id, movie_index_to_id

user_item_matrix, user_id_to_index, movie_id_to_index, user_index_to_id, movie_index_to_id = create_user_item_matrix(movie_ratings)

# Find similar movies using KNN
def find_similar_movies(movie_id, user_item_matrix, k, metric='cosine', show_distance=False):
    similar_movie_indices = []

    movie_index = movie_id_to_index[movie_id]
    movie_vector = user_item_matrix[movie_index]

    k_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    k_neighbors.fit(user_item_matrix)

    movie_vector = movie_vector.reshape(1, -1)
    distances, indices = k_neighbors.kneighbors(movie_vector, return_distance=True)

    for i in range(1, k+1):
        similar_movie_index = indices[0][i]
        similar_movie_indices.append(movie_index_to_id[similar_movie_index])

    return similar_movie_indices

movie_titles = dict(zip(movie_metadata['movieId'], movie_metadata['title']))

target_movie_id = 3
similar_movie_ids = find_similar_movies(target_movie_id, user_item_matrix, k=10)
target_movie_title = movie_titles[target_movie_id]

print(f"\nSince you watched '{target_movie_title}':")
for movie_id in similar_movie_ids:
    print(movie_titles.get(movie_id, "Unknown"))