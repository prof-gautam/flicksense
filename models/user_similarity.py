# user_similarity.py

import pandas as pd
import numpy as np

class UserSimilarityModel:
    def __init__(self, df_movies, df_ratings):
        self.df_movies = df_movies
        self.movie_data = df_ratings.pivot_table(index='userId', columns='movieId', values='rating')

    @staticmethod
    def calculate_cosine_similarity(x1, x2):
        numerator = np.dot(x1, x2)
        denominator = np.sqrt(np.dot(x1, x1) * np.dot(x2, x2))
        return numerator / denominator if denominator else 0

    def recommend_movie_by_similar_user(self, user_id, n=10):
        similarities = {}
        for user in self.movie_data.index:
            if user != user_id:
                sim = self.calculate_cosine_similarity(np.nan_to_num(self.movie_data.loc[user_id]), np.nan_to_num(self.movie_data.loc[user]))
                similarities[user] = sim
        sorted_users = sorted(similarities, key=similarities.get, reverse=True)[:n]
        recommend_movie_ids = []
        for user in sorted_users:
            recommend_movie_ids.extend(self.movie_data.loc[user].sort_values(ascending=False).head(n).index.tolist())
        recommend_movie_ids = list(set(recommend_movie_ids))[:n]
        recommend_movies = self.df_movies[self.df_movies['movieId'].isin(recommend_movie_ids)]['title'].tolist()
        return recommend_movies
