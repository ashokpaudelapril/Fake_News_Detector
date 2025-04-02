# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('fake_news_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create Flask app
app = Flask(__name__)

# Define a route to classify news
@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from the request
    news_text = request.json['text']
    
    # Convert the text into features using the saved TF-IDF vectorizer
    text_vector = vectorizer.transform([news_text])
    
    # Get the prediction from the model
    prediction = model.predict(text_vector)
    
    # Convert prediction to a human-readable format (0: fake, 1: real)
    result = "real" if prediction[0] == 1 else "fake"
    
    # Return the result as a JSON response
    return jsonify({"prediction": result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
