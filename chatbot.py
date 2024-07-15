
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)

# Load intents data
with open('intents.json', 'r') as f:
    data = json.load(f)

# Flatten patterns into individual patterns
patterns = []
tags = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
        responses.append(intent['responses'])

# Prepare data into a DataFrame
df = pd.DataFrame({'patterns': patterns, 'tag': tags, 'responses': responses})

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['patterns'])

# Train a Support Vector Machine (SVM) classifier
model = SVC()
model.fit(X, df['tag'])

# Global variables to store previous and current intents
previous_intent = None
current_intent = None

# Function to predict intents based on user input
def predict_intent(user_input):
    user_input_vec = vectorizer.transform([user_input])
    intent = model.predict(user_input_vec)[0]
    return intent

# Function to generate a response based on the predicted intent
def generate_response(intent):
    data = df[df['tag'] == intent]
    if data.empty:
        return "Sorry, I couldn't find a response for that intent."
    
    responses = data.iloc[0]['responses']
    if isinstance(responses, list):
        return random.choice(responses)
    else:
        return "Sorry, there was an issue with generating a response."

# Function to write data to data.json and trigger recommendation.py
def write_data_and_run_recommendation(previous_intent, selected_answer):
    data_to_write = {
        'previous_intent': previous_intent,
        'selected_answer': selected_answer
    }
    
    # Clear existing content and write new data
    with open('data.json', 'w') as file:
        json.dump(data_to_write, file, indent=4)
    
    # Run recommendation.py
    os.system('python recommendation.py')

# Route to serve index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting intents and generating responses
@app.route('/predict', methods=['POST'])
def predict():
    global previous_intent, current_intent
    
    data = request.get_json()
    user_input = data['user_input']
    current_intent = predict_intent(user_input)
    response = generate_response(current_intent)
    
    # Debugging: Log intent and selected answer
    print(f"Previous Intent: {previous_intent}")
    print(f"Predicted Intent: {current_intent}")
    selected_answer = data.get('selected_answer', None)
    print(f"Selected Answer: {selected_answer}")
    
    # Write data to data.json and trigger recommendation.py
    if selected_answer:
        write_data_and_run_recommendation(previous_intent, selected_answer)
    
    # Update the previous intent
    previous_intent = current_intent
    
    # Prepare recommended clothes data (replace with actual recommendation data)
    recommended_clothes = [
        {
            'image_url': 'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=500&auto=format&fit=crop&q=60',
            'weather': 'Sunny',
            'Occasion': 'Casual',
            'material': 'Cotton',
            'Suitability': 'fit'
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1516647982-dca2fef03961?w=500&auto=format&fit=crop&q=60',
            'weather': 'Rainy',
            'Occasion': 'Formal',
            'material': 'Wool',
            'Suitability': 'fit'
        }
        # Add more recommended clothes as needed
    ]
    
    # Construct JSON response with recommended clothes
    response_data = {
        'message': response,
        'recommended_clothes': recommended_clothes
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

