
# Flicksense: A Movielens Dataset-Based Movie Recommender

## Overview

Flicksense is a movie recommendation platform that harnesses the power of the Movielens dataset to provide curated cinematic suggestions to users. With a trio of models, the platform offers various lenses through which preferences can be matched, ensuring that movie enthusiasts can discover and enjoy titles aligned with their tastes. This system features an intuitive API constructed with FastAPI for streamlined user interaction.

## Project Structure

Flicksense/
│
├── data/ # Contains datasets for recommendation logic.
│ ├── links.csv # External identifiers for movie linkage.
│ ├── movies.csv # Movie titles and associated genres.
│ ├── ratings.csv # User-movie rating interactions.
│ └── tags.csv # User-generated metadata for movies.
│
├── models/ # Modules for each recommendation approach.
│ ├──  **init** .py # Designates the directory as a package.
│ ├── collaborative_filtering.py # Model for item-to-item recommendations.
│ ├── content_based.py # Recommends based on content similarity.
│ └── user_similarity.py # Personalized suggestions through user similarity.
│
├── notebooks/ # Jupyter notebooks for data exploration and visuals.
│ └── data_analysis_and_visualization.ipynb
│
├── main.py # Entry point to the FastAPI application.
└── requirements.txt # Project dependencies.


## Models Description

- **Item-to-Item Collaborative Filtering (`collaborative_filtering.py`)**:

  - This model provides recommendations by identifying similar movies based on a matrix of user ratings.
  - Utilizes a K-Nearest Neighbors algorithm trained on a sparse matrix of ratings to find the nearest neighbor movies.
- **Content-Based Filtering (`content_based.py`)**:

  - Offers suggestions by analyzing the content associated with the user's preferences, particularly focusing on movie genres and titles.
  - Employs TF-IDF Vectorization to transform text data into a feature space and computes similarity scores.
- **User Similarity (`user_similarity.py`)**:

  - Generates personalized movie recommendations by finding and analyzing users with similar viewing patterns and preferences.
  - Applies cosine similarity to user-rating vectors to determine users with tastes similar to a given user, then suggests movies that these similar users have rated highly.

## APIs

Flicksense's API endpoints are as follows:

- `/title-item-based-cf/{movie_name}`: Retrieve movie recommendations based on a specific movie title.
- `/user-similarity-based/{user_id}`: Acquire movie recommendations tailored for a particular user ID.

## Setup & Installation

Follow these steps to set up Flicksense on your local machine:

1. Ensure you have Python 3.8+ and pip installed.
2. Clone this repository to your local machine.
3. Navigate to the Flicksense directory.
4. Install dependencies with `pip install -r requirements.txt`.
5. Launch the application with `uvicorn main:app --reload`.

## Exploratory Data Analysis and Visualization

Explore the `notebooks/` directory for Jupyter notebooks that perform data cleaning, exploratory analysis, and visual representation of the dataset.

## Data

The project uses the Movielens dataset, split into multiple CSV files that are preprocessed and fed into the recommendation models.

## Requirements

A `requirements.txt` file lists all necessary Python packages. Install these with `pip install -r requirements.txt`.

## Running the Application

Execute the following command in the terminal at the project root to start the FastAPI server:

```bash
uvicorn main:app --reload
```
