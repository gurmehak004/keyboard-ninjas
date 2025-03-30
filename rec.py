import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Load and prepare data
df = pd.read_csv('shopping-site/last lowercase with link.csv')

# Feature Engineering
def create_features(df):
    # Age compatibility
    df['age_range'] = df['max_age'] - df['min_age']
    df[['hobby', 'relationship', 'occasion']] = df[['hobby', 'relationship', 'occasion']].fillna('Unknown')

    # Convert gender to numerical (0: Male, 1: Female, 2: Unisex)
    df['gender_code'] = df['gender'].map({'Male':0, 'Female':1, 'Unisex':2}).fillna(-1)

    # Historical gift performance
    gift_stats = df.groupby('gift_id').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    df = df.merge(gift_stats, on='gift_id')
    return df

df = create_features(df)

# Define features and target
X = df[['gender_code', 'hobby', 'relationship', 'occasion',
        'min_age', 'max_age', 'age_range', 'avg_rating', 'rating_count']]
y = df['rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['hobby', 'relationship', 'occasion']),
        ('num', StandardScaler(),
         ['gender_code', 'min_age', 'max_age', 'age_range',
          'avg_rating', 'rating_count'])
    ])

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 5],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Best Model RMSE: {rmse:.3f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Recommendation System
class OptimizedGiftRecommender:
    def __init__(self, model, df):
        self.model = model
        self.df = df
        self.gift_pool = df.drop_duplicates('gift_id')

    def recommend(self, user_profile, top_n=5):
        # Create input features for all gifts
        input_data = self.gift_pool.copy()
        input_data['user_age'] = user_profile['age']
        input_data['user_gender'] = user_profile['gender']

        # Calculate age match
        input_data['age_match'] = input_data.apply(
            lambda x: 1 if (x['min_age'] <= user_profile['age'] <= x['max_age']) else 0, axis=1
        )

        # Prepare features
        features = input_data[[
            'gender_code', 'hobby', 'relationship', 'occasion',
            'min_age', 'max_age', 'age_range', 'avg_rating',
            'rating_count', 'age_match'
        ]]

        # Predict ratings
        predicted_ratings = self.model.predict(features)

        # Get top recommendations
        recommendations = input_data.assign(predicted_rating=predicted_ratings)
        return recommendations.sort_values('predicted_rating', ascending=False).head(top_n)

# Initialize recommender with best model
recommender = OptimizedGiftRecommender(best_model, df)

# Example usage
user_profile = {
    'gender': 'male',
    'age': 28,
    'hobby': 'travel',
    'relationship': 'Friend',
    'occasion': 'wedding'
}

recommendations = recommender.recommend(user_profile)
print("\nTop Recommendations:")
print(recommendations[['gift_id', 'gift', 'predicted_rating', 'link']])