# main.py

import uvicorn
from fastapi import FastAPI
import pandas as pd
from models.collaborative_filtering import CollaborativeFilteringModel
from models.user_similarity import UserSimilarityModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "chrome-extension://idgfmgbkobhkgodkhbafbmhpccilpaca"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure you're pointing to the correct paths for your CSV files
df_movies = pd.read_csv("data/movies.csv")
df_ratings = pd.read_csv("data/ratings.csv")

cf_model = CollaborativeFilteringModel(df_movies, df_ratings)
cf_model.fit()
us_model = UserSimilarityModel(df_movies, df_ratings)

@app.get("/title-item-based-cf/{movie}")
async def title_item_based_cf(movie: str):
    recommendations = cf_model.get_recommendation(movie)
    return {"recommendations": recommendations}

@app.get("/user-similarity-based/{user_id}")
async def user_similarity_based(user_id: int):
    recommendations = us_model.recommend_movie_by_similar_user(user_id)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
