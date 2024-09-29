import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib

# Load the evaluation data from the CSV file
df = pd.read_csv('evaluation_data.csv')

# Prepare the data
X = df['Original Question']
y = df['Evaluation']  # Evaluation labels
w = df['New Question']
z = df['Rating']      # Ratings

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(X, y, z, w, test_size=0.2, random_state=20)

# Create pipelines for both evaluation and rating
evaluation_pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
rating_pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Define the parameter grid for tuning
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5]
}

# Use GridSearchCV for hyperparameter tuning on the evaluation model
grid_eval = GridSearchCV(evaluation_pipeline, param_grid, cv=2)  # Reduced cv to 2
grid_eval.fit(X_train, y_train)

# Save the best evaluation model
best_eval_model = grid_eval.best_estimator_
joblib.dump(best_eval_model, 'best_evaluation_model.joblib')

# Train the rating model using the original questions
rating_pipeline.fit(X_train, z_train)
joblib.dump(rating_pipeline, 'rating_model.joblib')

print("Models trained and saved.")
print("*" * 80)

# Function to evaluate new questions
def evaluate_questions(new_questions):
    try:
        eval_model = joblib.load('best_evaluation_model.joblib')
        rate_model = joblib.load('rating_model.joblib')
        
        evaluations = eval_model.predict(new_questions)
        ratings = rate_model.predict(new_questions)
        
        return evaluations, ratings
    except Exception as e:
        print(f"Error during evaluation: {e}")

# Example usage
new_questions_to_evaluate = [
    "blood?",
    "what is your blood",
    "what is your pressure",
    "what is your blood pressure",
    "Can you tell me your blood pressure?",
    "when did you go to school?",
    "pain ?",
    "here ?",
    "age ?",
    "do you pain?",
    "How are you feeling today?",
    "Why are you here?",
    "Are the symptoms constant or do they come and go?",
    "Pain?",
    "I don't know.",
    "What are your symptoms?",
    "Can you describe your pain?",
    "When did your symptoms start?",
    "Have you taken any medication?",
    "Are you experiencing any other issues?", "Why are you here today?",
    "How do you feel?",
    "When did you feel that?",
    "Are the symptoms constant or do they come and go?",
    "On a scale of 1 to 10, how severe is the pain/discomfort?",
    "Have you noticed anything that makes the symptoms better or worse?",
    "Do you have any pre-existing medical conditions?",
    "Are you currently taking any medications or supplements?",
    "Have you had any recent surgeries or hospitalizations?",
    "Do you have any known allergies?",
    "How often do you exercise?",
    "What is your diet like?",
    "Do you smoke or drink alcohol? If so, how much and how often?",
    "How many hours of sleep do you get on average?",
    "Does anyone in your family have a history of chronic conditions like diabetes, heart disease, or cancer?",
    "Are there any genetic conditions that run in your family?",
    "How have you been feeling emotionally?",
    "Have you experienced any recent changes in mood or behavior?",
    "Are you experiencing any stress or anxiety?",
    "Have you had a recent fever or chills?",
    "Have you experienced any unexplained weight loss or gain?",
    "Are you having any trouble with mobility or balance?",
    "Have you noticed any changes in your vision or hearing?",
    "Do you have any issues with digestion or bowel movements?",
    "Are you experiencing any problems with your breathing or chest pain?",
    "What is your favorite color?",
    "If you could travel anywhere, where would you go?",
    "How do you usually spend your weekends?",
    "Do you have any hobbies or interests?",
    "What's your favorite movie or book?",
    "How do you feel about pets? Do you have any?",
    "Do you think it's important to see a doctor regularly?",
    "What do you do to relax or unwind?",
    "Have you tried any home remedies for your symptoms?",
    "What was the last thing that made you laugh?",
    "Have you experienced any minor aches or pains recently?"
]

evaluations, ratings = evaluate_questions(new_questions_to_evaluate)
for question, evaluation, rating in zip(new_questions_to_evaluate, evaluations, ratings):
    print(f"Question: '{question}' ")
    print(f" Evaluation: {evaluation}")
    print(f"Rating: {rating}")
    print("-" * 80)
