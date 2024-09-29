import pandas as pd
from sklearn.pipeline import make_pipeline
import joblib
from sentence_transformers import SentenceTransformer, util

# Load the trained KNeighborsClassifier model
model = joblib.load('best_evaluation_model.joblib')

# Load a pre-trained model for semantic similarity



# Example new questions to evaluate
new_questions = [
    "pain?",
    "How do you feel?",
    "When did you feel that?",
]

# Function to evaluate new questions using the KNeighborsClassifier model
def evaluate_questions(new_questions):
    evaluations = model.predict(new_questions)
    
    # Create a DataFrame for formatted output
    results = pd.DataFrame({
        'New Question': new_questions,
        'Evaluation': evaluations
    })
    
    return results

# Get evaluations
evaluations_df = evaluate_questions(new_questions)


for index, row in evaluations_df.iterrows():
    print(f"Question: '{row['New Question']}' - Evaluation: {row['Evaluation']}")

