from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Load dataset
data = pd.read_csv('data.csv')

# Recommendation function
def recommend_products_based_on_preferences(weather, occasion, material):
    filtered_data = data[
        (data['weather'] == weather) &
        (data['occasion'] == occasion) &
        (data['material'] == material)
    ]
    return filtered_data[['image_path', 'weather', 'occasion', 'material']].to_dict('records')

# Endpoint to serve your index.html file
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Chatbot endpoint to get user preferences and return recommendations
@app.route('/chatbot', methods=['POST'])
def chatbot():
    req_data = request.get_json()
    weather = req_data.get('weather')
    occasion = req_data.get('occasion')
    material = req_data.get('material')

    recommendations = recommend_products_based_on_preferences(weather, occasion, material)
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
