import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load the evaluation data from the CSV file
df = pd.read_csv('evaluation_data.csv')

# Prepare the data
X = df['New Question']
y = df['Evaluation']  # You can choose to encode ratings if necessary
z = df ['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test ,z_train , z_test = train_test_split(X, y,z, test_size=0.2, random_state=42)

# Create a pipeline that combines a vectorizer and a classifier
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(X_train, y_train,z_train)

# Save the trained model
joblib.dump(model, 'question_evaluation_model.joblib')

print("Model trained and saved as 'question_evaluation_model.joblib'.")

# Function to evaluate new questions
def evaluate_questions(new_questions):
    # Load the trained model
    model = joblib.load('question_evaluation_model.joblib')
    
    # Make predictions on new questions
    predictions = model.predict(new_questions)
    
    return predictions

# Example usage:
new_questions_to_evaluate = [
    "How are you feeling today?",
    "Have you experienced any pain recently?"
]

evaluations = evaluate_questions(new_questions_to_evaluate)
for question, evaluation in zip(new_questions_to_evaluate, evaluations):
    print(f"Question: '{question}' - Evaluation: {evaluation}")