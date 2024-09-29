import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load the evaluation data from the CSV file
df = pd.read_csv('evaluation_data.csv')

# Prepare the data
X = df['Original Question']
y = df['Evaluation']  # Evaluation labels
z = df['Rating']      # Ratings
w = df['New Question']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, z_train, z_test,w_train,w_test = train_test_split(X, y, z,w ,test_size=0.2, random_state=42)

# Create pipelines for both evaluation and rating
evaluation_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
rating_model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the models
evaluation_model.fit(X_train, y_train)
rating_model.fit(w_train, z_train)

# Save the trained models
joblib.dump(evaluation_model, 'evaluation_model.joblib')
joblib.dump(rating_model, 'rating_model.joblib')

print("Models trained and saved.")

# Function to evaluate new questions
def evaluate_questions(new_questions):
    # Load the trained models
    eval_model = joblib.load('evaluation_model.joblib')
    rate_model = joblib.load('rating_model.joblib')
    
    # Make predictions on new questions
    evaluations = eval_model.predict(new_questions)
    ratings = rate_model.predict(new_questions)
    
    return evaluations, ratings

# Example usage:
new_questions_to_evaluate = [
    "How are you feeling today?",
    "why are you here?",
    "Are the symptoms constant or do they come and go?",
    "Pain?"
]

evaluations, ratings = evaluate_questions(new_questions_to_evaluate)
for question, evaluation, rating in zip(new_questions_to_evaluate, evaluations, ratings):
    print(f"Question: '{question}' - Evaluation: {evaluation}, Rating: {rating} ")
