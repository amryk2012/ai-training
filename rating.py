# Define the reference questions
reference_questions = [
    "What brings you in today?",
    "Can you describe your symptoms?",
    "When did the symptoms start?",
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
    "Are you experiencing any problems with your breathing or chest pain?"
]

# Function to evaluate a list of questions against the reference list
def evaluate_questions(new_questions):
    evaluations = {}

    for question in new_questions:
        evaluation = ""
        score = 0

        # Check for similarities and evaluate the question
        if question.lower() in [q.lower() for q in reference_questions]:
            score = 10
            evaluation = "This question is an exact match with the reference list."
        elif any(ref.lower() in question.lower() for ref in reference_questions):
            score = 8
            evaluation = "This question is related to the reference questions, but not an exact match."
        elif "feel" in question.lower() or "how" in question.lower():
            score = 6
            evaluation = "This question is more general and invites a broader response, lacking specific details about symptoms."
        elif "when" in question.lower() or "why" in question.lower():
            score = 5
            evaluation = "This question is somewhat relevant but vague, as it does not specify what is being referred to."
        else:
            score = 2
            evaluation = "This question is not relevant to the context of the reference questions."

        evaluations[question] = {
            "evaluation": evaluation,
            "score": score
        }

    return evaluations

# Example new questions to evaluate
new_questions = [
    " who are you?",
    "How do you feel?",
    "When did you feel that?"
]

# Evaluate the new questions
evaluated_results = evaluate_questions(new_questions)

# Print the evaluations in the desired format
print("Based on the provided list of questions, here's an evaluation and rating of the new questions:\n")

for question, result in evaluated_results.items():
    print(f"{question}\n")
    print(f"Evaluation: {result['evaluation']}")
    print(f"Rating: {result['score']}/10 ({'Very relevant' if result['score'] >= 8 else 'Relevant but lacks specificity' if result['score'] >= 5 else 'Somewhat relevant' if result['score'] >= 3 else 'Not relevant'})\n")

# Summary of Ratings
print("Summary of Ratings")
for question, result in evaluated_results.items():
    print(f"{question}: {result['score']}/10")
    
print("\nThese ratings reflect how closely the new questions align with the clarity and specificity needed in a medical context.")
