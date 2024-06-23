import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample dataset
data = {
    'User ID': [1, 1, 2, 2, 3, 3, 3],
    'Movie ID': [1, 2, 1, 3, 2, 3, 4],
    'Rating': [5, 3, 4, 5, 2, 4, 1]
}

df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot(index='User ID', columns='Movie ID', values='Rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Debugging: Print user similarity matrix
print("User Similarity Matrix:")
print(user_similarity_df)

# Predict ratings
def predict_ratings(user_id, user_item_matrix, user_similarity_df):
    similar_users = user_similarity_df.loc[user_id].drop(user_id)
    user_ratings = user_item_matrix.loc[user_id]
    
    predicted_ratings = pd.Series(index=user_item_matrix.columns)
    for movie in user_item_matrix.columns:
        if user_ratings[movie] == 0:
            # Extract ratings for the current movie from similar users
            ratings = user_item_matrix.loc[similar_users.index, movie]
            # Compute weighted sum
            numerator = np.dot(similar_users.values, ratings.values)
            denominator = similar_users.sum()
            if denominator == 0:
                predicted_ratings[movie] = user_item_matrix[movie].mean()  # Fallback to average rating for the movie
            else:
                predicted_ratings[movie] = numerator / denominator
        else:
            predicted_ratings[movie] = user_ratings[movie]
    
    # Debugging: Print predicted ratings for the user
    print("Predicted Ratings for User {}:".format(user_id))
    print(predicted_ratings)
    
    return predicted_ratings

user_id = 1
predicted_ratings = predict_ratings(user_id, user_item_matrix, user_similarity_df)

# Recommend movies
def recommend_movies(user_id, predicted_ratings, num_recommendations=3):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = predicted_ratings[unrated_movies].dropna().sort_values(ascending=False).head(num_recommendations)
    
    # Debugging: Print recommendations for the user
    print("Recommendations for User {}:".format(user_id))
    print(recommendations)
    
    return recommendations.index.tolist()

recommendations = recommend_movies(user_id, predicted_ratings)
print("Recommended Movies for User {}: {}".format(user_id, recommendations))