import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(
    X, y, z, w, test_size=0.2, random_state=42)

# Create pipelines for both evaluation and rating with RandomForestClassifier
evaluation_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),  # Include unigrams and bigrams
    RandomForestClassifier(random_state=42)
)

rating_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),  # Include unigrams and bigrams
    RandomForestClassifier(random_state=42)
)

# Train the models
evaluation_pipeline.fit(X_train, y_train)
rating_pipeline.fit(w_train, z_train)

# Save the trained models
joblib.dump(evaluation_pipeline, 'evaluation_model.joblib')
joblib.dump(rating_pipeline, 'rating_model.joblib')

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
    "Why are you here?",
    "Are the symptoms constant or do they come and go?",
    "Pain?",
    "I don't know.",
    "What are your symptoms?",
    "Can you describe your pain?",
    "When did your symptoms start?",
    "Have you taken any medication?",
    "Are you experiencing any other issues?"
]

evaluations, ratings = evaluate_questions(new_questions_to_evaluate)
for question, evaluation, rating in zip(new_questions_to_evaluate, evaluations, ratings):
    print(f"Question: '{question}' - Evaluation: {evaluation}, Rating: {rating}")
