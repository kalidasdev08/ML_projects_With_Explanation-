# Movie Recommendation System

A comprehensive movie recommendation system built with **Collaborative Filtering** and **Matrix Factorization (SVD)**. This system recommends movies to users based on three key factors:

1. **User History** - Movies the user has watched and rated
2. **Similar Users** - Collaborative filtering finding users with similar preferences
3. **Movie Genre** - Content-based filtering using movie genres

## 🎯 Features

- **Matrix Factorization (SVD)**: Uses Truncated SVD to learn latent factors for users and movies
- **Collaborative Filtering**: Finds similar users based on rating patterns
- **Content-Based Filtering**: Uses genre information to recommend similar movies
- **Hybrid Recommendations**: Combines all three methods for better recommendations

## 🏗️ Architecture

### ML Techniques Used

1. **Matrix Factorization (SVD)**
   - Decomposes the user-movie rating matrix into user and movie latent factors
   - Captures hidden patterns in user preferences
   - Uses 50 latent factors by default

2. **Collaborative Filtering (User-Based)**
   - Finds similar users using cosine similarity on user factors
   - Recommends movies that similar users have rated highly
   - Uses top 20 similar users for recommendations

3. **Content-Based Filtering**
   - Creates multi-hot encoding for movie genres
   - Computes genre similarity between movies
   - Builds user genre preference profile from rated movies

## 📁 Project Structure

```
Movie Recommendation System/
├── app.py                    # Flask web application
├── train_model.py            # Model training and recommendation logic
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── templates/
│   └── index.html           # Web interface
└── models/
    └── movie_recommender.pkl # Trained model
```

## 🚀 Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   cd "Movie Recommendation System"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and visit:
   ```
   http://localhost:5000
   ```

## 💻 Usage

### Web Interface

1. Select a user from the dropdown menu
2. View personalized recommendations in different categories:
   - **All Recommendations**: Combined recommendations from all methods
   - **Based on Your History**: SVD-based predictions
   - **Similar Users**: Movies watched by similar users
   - **By Genre**: Genre-preference based recommendations

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/users` | List all users |
| GET | `/api/movies` | List all movies |
| GET | `/api/user/<id>/ratings` | Get user's rated movies |
| POST | `/api/recommend` | Get all recommendations |
| POST | `/api/recommend/history` | Get history-based recommendations |
| POST | `/api/recommend/similar-users` | Get similar user recommendations |
| POST | `/api/recommend/genre` | Get genre-based recommendations |
| GET | `/api/similar-users/<id>` | Find similar users |
| GET | `/api/stats` | Model statistics |

### Example API Usage

```python
import requests

# Get recommendations
response = requests.post('http://localhost:5000/api/recommend', json={
    'user_id': 'user_1',
    'n_recommendations': 10
})
recommendations = response.json()
```

## 🔧 Configuration

You can customize the recommender by modifying parameters in `train_model.py`:

```python
recommender = MovieRecommender(
    n_factors=50,      # Number of latent factors for SVD
    n_neighbors=20    # Number of similar users to consider
)
```

## 📊 Model Details

- **Dataset**: Synthetic movie ratings (100 users, 50 movies, 2000+ ratings)
- **Rating Scale**: 1-5 stars
- **Genres**: Action, Comedy, Drama, Horror, Sci-Fi, Romance, Thriller, Animation, Documentary, Fantasy
- **SVD Explained Variance**: ~70-80%

## 🎬 Sample Movies

The system includes popular movies from various genres:
- Action: The Dark Knight, Inception, Avengers
- Sci-Fi: Interstellar, The Matrix, Gravity
- Drama: Forrest Gump, The Shawshank Redemption
- Comedy: 3 Idiots, Frozen, Toy Story
- And many more...

## 🧠 How It Works

### 1. User History-Based Recommendations
- Uses SVD to predict ratings for unrated movies
- Ranks movies by predicted rating
- Based on the user's rating patterns

### 2. Similar Users Recommendations
- Finds users with similar movie preferences
- Uses cosine similarity on user latent factors
- Recommends movies rated highly by similar users

### 3. Genre-Based Recommendations
- Builds user genre preference profile from rated movies
- Scores movies based on genre match
- Weights genres by user's rating history

## 🛠️ Technologies Used

- **Python 3.x**
- **Flask** - Web framework
- **NumPy/Pandas** - Data processing
- **Scikit-learn** - SVD and cosine similarity
- **SciPy** - Sparse matrix operations

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created as a demonstration of recommender systems using collaborative filtering and matrix factorization.
