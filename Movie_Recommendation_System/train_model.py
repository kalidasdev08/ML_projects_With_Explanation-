"""
Movie Recommendation System
Using Collaborative Filtering and Matrix Factorization (SVD)

This module implements a movie recommendation system based on:
1. User history - movies the user has watched and rated
2. Similar users - collaborative filtering finding similar users
3. Movie genre - content-based filtering using movie genres
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
import os
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class MovieRecommender:
    """Movie Recommendation System using Collaborative Filtering and Matrix Factorization"""
    
    def __init__(self, n_factors=50, n_neighbors=20):
        """
        Initialize the recommender system
        
        Args:
            n_factors: Number of latent factors for SVD
            n_neighbors: Number of similar users to consider
        """
        self.n_factors = n_factors
        self.n_neighbors = n_neighbors
        self.user_movie_matrix = None
        self.movie_user_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.movie_titles = None
        self.movie_genres = None
        self.user_idx_map = None
        self.idx_user_map = None
        self.movie_idx_map = None
        self.idx_movie_map = None
        self.svd = None
        self.user_factors = None
        self.movie_factors = None
        self.genre_similarity = None
        self.mean_ratings = None
        self.global_mean = None
        
    def generate_sample_data(self, n_users=100, n_movies=50, n_ratings=2000):
        """
        Load real movie ratings dataset
        
        Args:
            Path to the CSV file (default: movie_ratings.csv in same directory)
        """
        print("Loading real dataset from movie_ratings.csv...")
        
        # Load the CSV file
        import os
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movie_ratings.csv')
        
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}, generating synthetic data instead...")
            return self._generate_synthetic_data(n_users, n_movies, n_ratings)
        
        ratings_df = pd.read_csv(csv_path)
        
        print(f"Loaded dataset with {len(ratings_df)} ratings")
        print(f"Columns: {list(ratings_df.columns)}")
        
        # Get unique users and movies (keep as integers for matching with dataframe)
        self.user_ids = sorted(ratings_df['user_id'].unique())
        self.movie_ids = sorted(ratings_df['movie_id'].unique())
        
        print(f"Unique users: {len(self.user_ids)}")
        print(f"Unique movies: {len(self.movie_ids)}")
        
        # Create movie titles (using movie_id as title for real dataset)
        self.movie_titles = {mid: f"Movie {mid}" for mid in self.movie_ids}
        
        # Create genre mapping (one genre per movie from the dataset)
        movie_genres_df = ratings_df.drop_duplicates('movie_id')[['movie_id', 'genre']]
        self.movie_genres = {}
        for _, row in movie_genres_df.iterrows():
            mid = row['movie_id']
            title = self.movie_titles[mid]
            self.movie_genres[title] = [row['genre']]
        
        # Add any missing movies with default genre
        for mid in self.movie_ids:
            title = self.movie_titles[mid]
            if title not in self.movie_genres:
                self.movie_genres[title] = ['Drama']  # default genre
        
        # Create mappings (use integers)
        self.user_idx_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.idx_user_map = {idx: uid for uid, idx in self.user_idx_map.items()}
        self.movie_idx_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_movie_map = {idx: mid for mid, idx in self.movie_idx_map.items()}
        
        print(f"Dataset loaded successfully!")
        return ratings_df
    
    def _generate_synthetic_data(self, n_users=100, n_movies=50, n_ratings=2000):
        """
        Generate synthetic movie rating data for demonstration
        
        Args:
            n_users: Number of users
            n_movies: Number of movies
            n_ratings: Number of ratings to generate
        """
        print(f"Generating sample data with {n_users} users, {n_movies} movies...")
        
        # Movie titles and genres
        genres_list = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 
                       'Romance', 'Thriller', 'Animation', 'Documentary', 'Fantasy']
        
        movie_titles = [
            "The Dark Knight", "Inception", "Interstellar", "The Matrix", "Pulp Fiction",
            "Forrest Gump", "The Shawshank Redemption", "Fight Club", "The Godfather",
            "Titanic", "Avatar", "The Avengers", "Frozen", "Toy Story", "The Lion King",
            "Joker", "Parasite", "Bahubali", "Dangal", "3 Idiots",
            "The Conjuring", "IT", "Get Out", "A Quiet Place", "Train to Busan",
            "La La Land", "The Notebook", "Pride and Prejudice", "Crazy Rich Asians",
            "The Martian", "Gravity", "The Prestige", "Memento", "Gladiator",
            "The Dark Knight Rises", "Avengers Endgame", "Spider-Man", "Black Panther",
            "Wonder Woman", "Aquaman", "Joker", "Deadpool", "Guardians of the Galaxy",
            "Thor", "Iron Man", "Captain America", "Doctor Strange", "Ant-Man",
            "The Flash", "Batman"
        ]
        
        movie_titles = movie_titles[:n_movies]
        
        # Assign random genres to movies
        movie_genres = {}
        for movie in movie_titles:
            num_genres = random.randint(1, 3)
            movie_genres[movie] = random.sample(genres_list, num_genres)
        
        self.movie_titles = movie_titles
        self.movie_genres = movie_genres
        
        # Generate user IDs
        self.user_ids = [f"user_{i}" for i in range(1, n_users + 1)]
        
        # Generate movie IDs
        self.movie_ids = [f"movie_{i}" for i in range(1, n_movies + 1)]
        
        # Create mappings
        self.user_idx_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.idx_user_map = {idx: uid for uid, idx in self.user_idx_map.items()}
        self.movie_idx_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_movie_map = {idx: mid for mid, idx in self.movie_idx_map.items()}
        
        # Generate random ratings
        ratings = []
        for _ in range(n_ratings):
            user_id = random.choice(self.user_ids)
            movie_id = random.choice(self.movie_ids)
            # Ratings tend to be higher (more likely to watch movies they like)
            rating = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.1, 0.2, 0.35, 0.3]
            )[0]
            ratings.append((user_id, movie_id, rating))
        
        # Remove duplicates (keep last rating)
        ratings = list(set(ratings))
        
        # Create DataFrame
        ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'])
        
        print(f"Generated {len(ratings_df)} ratings")
        return ratings_df
    
    def create_user_movie_matrix(self, ratings_df):
        """
        Create user-movie rating matrix
        
        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns
        """
        print("Creating user-movie matrix...")
        
        # Create matrix (keep as integers)
        self.user_movie_matrix = pd.pivot_table(
            ratings_df,
            values='rating',
            index='user_id',
            columns='movie_id',
            fill_value=0
        )
        
        # Ensure all movies are present
        for movie_id in self.movie_ids:
            if movie_id not in self.user_movie_matrix.columns:
                self.user_movie_matrix[movie_id] = 0
        
        # Reorder columns
        self.user_movie_matrix = self.user_movie_matrix[self.movie_ids]
        
        # Store movie_user matrix (transpose)
        self.movie_user_matrix = self.user_movie_matrix.T
        
        print(f"User-Movie matrix shape: {self.user_movie_matrix.shape}")
        print(f"Total ratings in matrix: {(self.user_movie_matrix > 0).sum().sum()}")
        return self.user_movie_matrix
    
    def compute_genre_similarity(self):
        """
        Compute genre-based similarity between movies for content-based filtering
        """
        print("Computing genre similarity...")
        
        n_movies = len(self.movie_ids)
        
        # Create genre vectors (multi-hot encoding)
        all_genres = list(set(genre for genres in self.movie_genres.values() for genre in genres))
        
        genre_vectors = np.zeros((n_movies, len(all_genres)))
        
        for i, movie in enumerate(self.movie_titles):
            for genre in self.movie_genres.get(movie, []):
                if genre in all_genres:
                    genre_vectors[i, all_genres.index(genre)] = 1
        
        # Compute cosine similarity
        self.genre_similarity = cosine_similarity(genre_vectors)
        
        print(f"Genre similarity matrix shape: {self.genre_similarity.shape}")
        return self.genre_similarity
    
    def fit(self, ratings_df=None):
        """
        Train the recommendation model using SVD (Matrix Factorization)
        
        Args:
            ratings_df: DataFrame with user_id, movie_id, rating columns
        """
        if ratings_df is None:
            ratings_df = self.generate_sample_data()
        
        # Create user-movie matrix
        self.create_user_movie_matrix(ratings_df)
        
        # Compute mean ratings per user
        self.mean_ratings = ratings_df.groupby('user_id')['rating'].mean()
        self.global_mean = ratings_df['rating'].mean()
        
        # Compute genre similarity
        self.compute_genre_similarity()
        
        # Apply SVD for matrix factorization using numpy's SVD
        print(f"Training SVD with {self.n_factors} latent factors...")
        
        # Use the rating matrix for SVD
        R = self.user_movie_matrix.values.astype(float)
        
        # Center the ratings by subtracting user mean (for unrated, we use 0)
        # We need to handle this carefully for sparse matrices
        # For simplicity, we'll use the global mean centering
        
        # Apply SVD using numpy (more control over the decomposition)
        # R ≈ U * Σ * V^T
        # where:
        #   U (m x k): user latent factors
        #   Σ (k x k): diagonal matrix of singular values
        #   V^T (k x n): movie latent factors
        
        # Use sklearn's TruncatedSVD which is designed for this purpose
        # It finds the best approximation R ≈ U * Σ * V^T with k components
        
        self.svd = TruncatedSVD(n_components=min(self.n_factors, min(R.shape) - 1), random_state=42)
        
        # Fit on the user-movie matrix to get user factors
        # Returns: (n_users, n_components)
        self.user_factors = self.svd.fit_transform(R)
        
        # Get movie factors by transforming the transpose
        # V = (R^T * U * Σ^(-1))
        # Or equivalently: V = Σ * V^T
        self.movie_factors = self.svd.components_.T  # (n_movies, n_components)
        
        # The singular values
        singular_values = self.svd.singular_values_
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Movie factors shape: {self.movie_factors.shape}")
        
        # Calculate explained variance
        explained_var = sum(self.svd.explained_variance_ratio_) * 100
        print(f"Total explained variance: {explained_var:.2f}%")
        
        print("Training complete!")
        return self
    
    def find_similar_users(self, user_id, n_neighbors=None):
        """
        Find similar users using collaborative filtering
        
        Args:
            user_id: The user ID to find similar users for
            n_neighbors: Number of similar users to return
            
        Returns:
            List of (similar_user_id, similarity_score) tuples
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        if user_id not in self.user_idx_map:
            return []
        
        user_idx = self.user_idx_map[user_id]
        
        # Compute user similarity using user factors
        user_vector = self.user_factors[user_idx].reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.user_factors)[0]
        
        # Get top similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        similar_users = []
        for idx in similar_indices:
            if idx != user_idx:
                similar_user_id = self.idx_user_map[idx]
                similarity = similarities[idx]
                similar_users.append((similar_user_id, similarity))
                if len(similar_users) >= n_neighbors:
                    break
        
        return similar_users
    
    def get_user_rated_movies(self, user_id):
        """
        Get movies rated by a user
        
        Args:
            user_id: The user ID
            
        Returns:
            Dictionary of movie_id: rating
        """
        if user_id not in self.user_movie_matrix.index:
            return {}
        
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = {}
        
        for movie_id, rating in user_ratings.items():
            if rating > 0:
                rated_movies[movie_id] = rating
                
        return rated_movies
    
    def recommend_by_user_history(self, user_id, n_recommendations=5):
        """
        Recommend movies based on user's rating history
        
        Args:
            user_id: The user ID
            n_recommendations: Number of movies to recommend
            
        Returns:
            List of (movie_id, predicted_rating, reason) tuples
        """
        if user_id not in self.user_idx_map:
            return []
        
        user_idx = self.user_idx_map[user_id]
        
        # Get user's rating vector
        user_ratings = self.user_movie_matrix.loc[user_id].values
        
        # Predict ratings for all movies using SVD
        # user_factors: (n_users, n_factors), movie_factors: (n_movies, n_factors)
        # prediction = user_factors @ movie_factors.T
        predicted_ratings = np.dot(
            self.user_factors[user_idx],
            self.movie_factors.T
        )
        
        # Get movies user hasn't rated
        unrated_movies = []
        for i, movie_id in enumerate(self.movie_ids):
            if user_ratings[i] == 0:
                # Add small noise to break ties
                predicted = predicted_ratings[i]
                unrated_movies.append((movie_id, predicted, "Based on your rating pattern"))
        
        # Sort by predicted rating
        unrated_movies.sort(key=lambda x: x[1], reverse=True)
        
        return unrated_movies[:n_recommendations]
    
    def recommend_by_similar_users(self, user_id, n_recommendations=5):
        """
        Recommend movies based on what similar users have watched
        
        Args:
            user_id: The user ID
            n_recommendations: Number of movies to recommend
            
        Returns:
            List of (movie_id, score, reason) tuples
        """
        similar_users = self.find_similar_users(user_id)
        
        if not similar_users:
            return []
        
        # Get movies watched by similar users
        user_id = self.user_idx_map[user_id]
        user_watched = set(self.user_movie_matrix.columns[self.user_movie_matrix.loc[self.idx_user_map[user_id]] > 0])
        
        # Weighted average of similar users' ratings
        movie_scores = {}
        
        for similar_user, similarity in similar_users:
            similar_user_ratings = self.user_movie_matrix.loc[similar_user]
            
            for movie_id, rating in similar_user_ratings.items():
                if rating > 0 and movie_id not in user_watched:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = {'weighted_sum': 0, 'weight_sum': 0}
                    
                    movie_scores[movie_id]['weighted_sum'] += similarity * rating
                    movie_scores[movie_id]['weight_sum'] += similarity
        
        # Calculate normalized scores
        recommendations = []
        for movie_id, scores in movie_scores.items():
            if scores['weight_sum'] > 0:
                normalized_score = scores['weighted_sum'] / scores['weight_sum']
                recommendations.append((movie_id, normalized_score, f"Users similar to you rated this highly"))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_by_genre(self, user_id, n_recommendations=5):
        """
        Recommend movies based on genre preferences
        
        Args:
            user_id: The user ID
            n_recommendations: Number of movies to recommend
            
        Returns:
            List of (movie_id, score, reason) tuples
        """
        # Get user's genre preferences from rated movies
        user_ratings = self.get_user_rated_movies(user_id)
        
        if not user_ratings:
            # If no history, recommend popular genres
            genre_scores = {'Action': 0.8, 'Sci-Fi': 0.7, 'Drama': 0.6}
        else:
            # Calculate genre preferences based on ratings
            genre_scores = {}
            
            for movie_id, rating in user_ratings.items():
                movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
                genres = self.movie_genres.get(movie_title, [])
                
                for genre in genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating)
            
            # Average genre preferences
            genre_scores = {g: np.mean(ratings) for g, ratings in genre_scores.items()}
        
        # Score all movies
        movie_scores = []
        user_watched = set(user_ratings.keys())
        
        for i, movie_id in enumerate(self.movie_ids):
            if movie_id in user_watched:
                continue
                
            movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
            genres = self.movie_genres.get(movie_title, [])
            
            if not genres:
                continue
                
            # Calculate score based on genre preferences
            score = sum(genre_scores.get(g, 0) for g in genres) / len(genres)
            
            movie_scores.append((movie_id, score, f"Matches your favorite genres: {', '.join(genres)}"))
        
        # Sort by score
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return movie_scores[:n_recommendations]
    
    def recommend(self, user_id, n_recommendations=10):
        """
        Get comprehensive recommendations combining all methods
        
        Args:
            user_id: The user ID
            n_recommendations: Number of movies to recommend
            
        Returns:
            Dictionary with recommendation categories
        """
        # Get recommendations from each method
        history_recs = self.recommend_by_user_history(user_id, n_recommendations)
        similar_users_recs = self.recommend_by_similar_users(user_id, n_recommendations)
        genre_recs = self.recommend_by_genre(user_id, n_recommendations)
        
        # Combine and deduplicate
        combined = {}
        
        for rec in history_recs:
            movie_id = rec[0]
            movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
            combined[movie_id] = {
                'movie_id': movie_id,
                'movie_title': movie_title,
                'genres': self.movie_genres.get(movie_title, []),
                'score': rec[1],
                'reason': rec[2]
            }
        
        for rec in similar_users_recs:
            movie_id = rec[0]
            movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
            if movie_id not in combined:
                combined[movie_id] = {
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'genres': self.movie_genres.get(movie_title, []),
                    'score': rec[1],
                    'reason': rec[2]
                }
            else:
                # Combine scores
                combined[movie_id]['score'] = (combined[movie_id]['score'] + rec[1]) / 2
                combined[movie_id]['reason'] += " | " + rec[2]
        
        for rec in genre_recs:
            movie_id = rec[0]
            movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
            if movie_id not in combined:
                combined[movie_id] = {
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'genres': self.movie_genres.get(movie_title, []),
                    'score': rec[1],
                    'reason': rec[2]
                }
            else:
                combined[movie_id]['score'] = (combined[movie_id]['score'] + rec[1]) / 2
                combined[movie_id]['reason'] += " | " + rec[2]
        
        # Sort by combined score
        recommendations = list(combined.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'user_id': user_id,
            'total_recommendations': len(recommendations),
            'recommendations': recommendations[:n_recommendations]
        }
    
    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        if movie_id not in self.movie_idx_map:
            return None
        
        movie_title = self.movie_titles.get(str(movie_id), f"Movie {movie_id}")
        return {
            'movie_id': movie_id,
            'title': movie_title,
            'genres': self.movie_genres.get(movie_title, [])
        }
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'n_factors': self.n_factors,
            'n_neighbors': self.n_neighbors,
            'user_ids': self.user_ids,
            'movie_ids': self.movie_ids,
            'movie_titles': self.movie_titles,
            'movie_genres': self.movie_genres,
            'user_idx_map': self.user_idx_map,
            'idx_user_map': self.idx_user_map,
            'movie_idx_map': self.movie_idx_map,
            'idx_movie_map': self.idx_movie_map,
            'user_factors': self.user_factors,
            'movie_factors': self.movie_factors,
            'genre_similarity': self.genre_similarity,
            'mean_ratings': self.mean_ratings,
            'global_mean': self.global_mean,
            'user_movie_matrix': self.user_movie_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_factors = model_data['n_factors']
        self.n_neighbors = model_data['n_neighbors']
        self.user_ids = model_data['user_ids']
        self.movie_ids = model_data['movie_ids']
        self.movie_titles = model_data['movie_titles']
        self.movie_genres = model_data['movie_genres']
        self.user_idx_map = model_data['user_idx_map']
        self.idx_user_map = model_data['idx_user_map']
        self.movie_idx_map = model_data['movie_idx_map']
        self.idx_movie_map = model_data['idx_movie_map']
        self.user_factors = model_data['user_factors']
        self.movie_factors = model_data['movie_factors']
        self.genre_similarity = model_data['genre_similarity']
        self.mean_ratings = model_data['mean_ratings']
        self.global_mean = model_data['global_mean']
        self.user_movie_matrix = model_data.get('user_movie_matrix')
        
        print(f"Model loaded from {filepath}")


def train_model(model_path='models/movie_recommender.pkl'):
    """Train and save the recommendation model"""
    # Create model instance
    recommender = MovieRecommender(n_factors=50, n_neighbors=20)
    
    # Train the model
    recommender.fit()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    recommender.save_model(model_path)
    
    return recommender


if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Test recommendations for a sample user
    test_user = "user_1"
    print(f"\n{'='*50}")
    print(f"Testing recommendations for {test_user}")
    print(f"{'='*50}")
    
    # Get user's rated movies
    rated = model.get_user_rated_movies(test_user)
    print(f"\nMovies rated by {test_user}:")
    for movie_id, rating in list(rated.items())[:5]:
        movie_title = model.movie_titles[model.movie_ids.index(movie_id)]
        genres = model.movie_genres.get(movie_title, [])
        print(f"  - {movie_title} ({rating}/5) - {', '.join(genres)}")
    
    # Get recommendations
    recommendations = model.recommend(test_user, n_recommendations=10)
    
    print(f"\nRecommended Movies for {test_user}:")
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"\n{i}. {rec['movie_title']}")
        print(f"   Genres: {', '.join(rec['genres'])}")
        print(f"   Score: {rec['score']:.2f}")
        print(f"   Reason: {rec['reason']}")
    
    # Find similar users
    similar_users = model.find_similar_users(test_user, n_neighbors=5)
    print(f"\n\nSimilar Users to {test_user}:")
    for user, similarity in similar_users:
        print(f"  - {user} (similarity: {similarity:.3f})")
