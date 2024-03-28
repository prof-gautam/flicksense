import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class CollaborativeFilteringModel:
    def __init__(self, df_movies, df_ratings):
        self.df_movies = df_movies
        if 'userId' not in df_ratings.columns:
            raise ValueError("DataFrame does not contain 'userId' column")
        self.csr_data, self.data_final = self.prepare_data(df_ratings)

    def prepare_data(self, df_ratings):
        pivot_data = pd.pivot_table(df_ratings, index='movieId', columns='userId', values='rating').fillna(0)
        csr_data = csr_matrix(pivot_data.values)
        data_final = pivot_data.reset_index()
        return csr_data, data_final

    def fit(self):
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        self.model.fit(self.csr_data)

    def get_recommendation(self, movie_name, n=10):
        # Convert the input movie name to lowercase
        movie_name = movie_name.lower()
        
        # Convert movie titles in the DataFrame to lowercase
        self.df_movies['title'] = self.df_movies['title'].str.lower()
        
        movie_list = self.df_movies[self.df_movies['title'].str.contains(movie_name, regex=False)]
        if not movie_list.empty:
            movie_idx = movie_list.iloc[0]['movieId']
            if movie_idx not in self.data_final['movieId'].values:
                return "Movie not found in the ratings data."
            movie_idx_row = self.data_final[self.data_final['movieId'] == movie_idx].index[0]
            distances, indices = self.model.kneighbors(self.csr_data[movie_idx_row], n_neighbors=n+1)
            rec_movie_indices = indices.squeeze().tolist()[1:]  # Exclude the movie itself
            
            # Convert recommended movie titles back to original case
            recommendations = [self.df_movies.iloc[idx]['title'].capitalize() for idx in rec_movie_indices]
            return recommendations
        else:
            return "No movies found. Please check your input."
