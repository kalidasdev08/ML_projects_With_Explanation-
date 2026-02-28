"""
Movie Recommendation System - Flask Web Application

A web application that recommends movies based on:
1. User history - movies the user has watched and rated
2. Similar users - collaborative filtering finding similar users  
3. Movie genre - content-based filtering using movie genres

ML Type: Recommender System
Skills: Collaborative Filtering, Matrix Factorization (SVD)
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import MovieRecommender
from movie_data import get_movie_info, MOVIE_DATABASE

app = Flask(__name__)

# Global model instance
recommender = None


def get_user_id(user_id_str):
    """Convert user_id string to appropriate type based on model"""
    if recommender is None:
        return user_id_str
    # Check if model uses integer IDs
    if recommender.user_ids and isinstance(recommender.user_ids[0], (int, np.integer)):
        try:
            return int(user_id_str)
        except:
            return user_id_str
    return user_id_str


def initialize_model():
    """Initialize or load the recommendation model"""
    global recommender
    
    model_path = 'Movie_Recommendation_System/models/movie_recommender.pkl'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        recommender = MovieRecommender()
        recommender.load_model(model_path)
    else:
        print("Training new model...")
        # Import and train
        from train_model import train_model
        os.makedirs('Movie_Recommendation_System/models', exist_ok=True)
        recommender = train_model(model_path)
    
    return recommender


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get list of all users"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Convert user IDs to strings for JSON serialization
    users = [str(u) for u in recommender.user_ids]
    return jsonify({'users': users})


@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get list of all movies with posters"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    movies = []
    for mid in recommender.movie_ids:
        movie_info = get_movie_info(mid)
        movies.append({
            'id': mid,
            'title': movie_info['title'],
            'poster': movie_info['poster'],
            'year': movie_info['year'],
            'genres': recommender.movie_genres.get(f"Movie {mid}", [])
        })
    
    return jsonify({'movies': movies})


@app.route('/api/user/<user_id>/ratings', methods=['GET'])
def get_user_ratings(user_id):
    """Get movies rated by a user"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    user_id = get_user_id(user_id)
    ratings = recommender.get_user_rated_movies(user_id)
    
    rated_movies = []
    for movie_id, rating in ratings.items():
        movie_info = get_movie_info(movie_id)
        rated_movies.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'poster': movie_info['poster'],
            'year': movie_info['year'],
            'rating': rating,
            'genres': recommender.movie_genres.get(f"Movie {movie_id}", [])
        })
    
    return jsonify({'user_id': user_id, 'ratings': rated_movies})


@app.route('/api/recommend', methods=['POST'])
def recommend_movies():
    """Get movie recommendations"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 10)
    
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    
    user_id = get_user_id(user_id)
    
    if user_id not in recommender.user_ids:
        return jsonify({'error': f'User {user_id} not found'}), 404
    
    try:
        recommendations = recommender.recommend(user_id, n_recommendations)
        # Add movie info to recommendations
        for rec in recommendations.get('recommendations', []):
            movie_info = get_movie_info(rec['movie_id'])
            rec['title'] = movie_info['title']
            rec['poster'] = movie_info['poster']
            rec['year'] = movie_info['year']
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend/history', methods=['POST'])
def recommend_by_history():
    """Get recommendations based on user history"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 5)
    
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    
    user_id = get_user_id(user_id)
    
    recommendations = recommender.recommend_by_user_history(user_id, n_recommendations)
    
    result = []
    for movie_id, score, reason in recommendations:
        movie_info = get_movie_info(movie_id)
        result.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'poster': movie_info['poster'],
            'year': movie_info['year'],
            'genres': recommender.movie_genres.get(f"Movie {movie_id}", []),
            'score': float(score),
            'reason': reason
        })
    
    return jsonify({'user_id': user_id, 'recommendations': result})


@app.route('/api/recommend/similar-users', methods=['POST'])
def recommend_by_similar_users():
    """Get recommendations based on similar users"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 5)
    
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    
    user_id = get_user_id(user_id)
    
    recommendations = recommender.recommend_by_similar_users(user_id, n_recommendations)
    
    result = []
    for movie_id, score, reason in recommendations:
        movie_info = get_movie_info(movie_id)
        result.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'poster': movie_info['poster'],
            'year': movie_info['year'],
            'genres': recommender.movie_genres.get(f"Movie {movie_id}", []),
            'score': float(score),
            'reason': reason
        })
    
    return jsonify({'user_id': user_id, 'recommendations': result})


@app.route('/api/recommend/genre', methods=['POST'])
def recommend_by_genre():
    """Get recommendations based on genre preferences"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 5)
    
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    
    user_id = get_user_id(user_id)
    
    recommendations = recommender.recommend_by_genre(user_id, n_recommendations)
    
    result = []
    for movie_id, score, reason in recommendations:
        movie_info = get_movie_info(movie_id)
        result.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'poster': movie_info['poster'],
            'year': movie_info['year'],
            'genres': recommender.movie_genres.get(f"Movie {movie_id}", []),
            'score': float(score),
            'reason': reason
        })
    
    return jsonify({'user_id': user_id, 'recommendations': result})


@app.route('/api/similar-users/<user_id>', methods=['GET'])
def get_similar_users(user_id):
    """Get similar users"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    user_id = get_user_id(user_id)
    
    similar_users = recommender.find_similar_users(user_id)
    
    result = []
    for user, similarity in similar_users:
        result.append({
            'user_id': user,
            'similarity': float(similarity)
        })
    
    return jsonify({'user_id': user_id, 'similar_users': result})


@app.route('/api/movie/<movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get movie details"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    movie_info = recommender.get_movie_info(movie_id)
    
    if movie_info is None:
        return jsonify({'error': 'Movie not found'}), 404
    
    return jsonify(movie_info)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'n_users': len(recommender.user_ids),
        'n_movies': len(recommender.movie_ids),
        'n_factors': recommender.n_factors,
        'n_neighbors': recommender.n_neighbors,
        'matrix_shape': recommender.user_movie_matrix.shape if recommender.user_movie_matrix is not None else None
    })


# Initialize model on startup
with app.app_context():
    initialize_model()


if __name__ == '__main__':
    print("=" * 60)
    print("Movie Recommendation System")
    print("=" * 60)
    print("\nEndpoints available:")
    print("  - GET  /                     : Main page")
    print("  - GET  /api/users            : List all users")
    print("  - GET  /api/movies           : List all movies")
    print("  - GET  /api/user/<id>/ratings: Get user's rated movies")
    print("  - POST /api/recommend        : Get recommendations (JSON: {user_id, n_recommendations})")
    print("  - POST /api/recommend/history    : Recommendations based on user history")
    print("  - POST /api/recommend/similar-users: Recommendations based on similar users")
    print("  - POST /api/recommend/genre      : Recommendations based on genre")
    print("  - GET  /api/similar-users/<id>   : Find similar users")
    print("  - GET  /api/movie/<id>      : Get movie details")
    print("  - GET  /api/stats           : Model statistics")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
