# app.py

import sys
import os
import pickle
from flask import Flask, request, jsonify, render_template
from utils import preprocess_text  # Import the same preprocessing function

# Ensure the app can find the utils.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

# Load Model and Vectorizer 
try:
    with open('./saved_models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
    
    with open('./saved_models/vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    print("Vectorizer loaded successfully.")

except FileNotFoundError:
    print("Error: Model or vectorizer files not found in ./saved_models/")
    print("Please run train.py first to create these files.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading files: {e}")
    sys.exit(1)


# Define Routes 

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives text input and returns a prediction."""
    try:
        # Get text from the POST request
        data = request.get_json(force=True)
        text = data['text']

        if not text.strip():
            return jsonify({'error': 'Input text is empty'}), 400

        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Vectorize the text
        vectorized_text = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)
        
        # Format the output
        label = 'Real' if prediction[0] == 1 else 'Fake'
        # Get the confidence score for the predicted class
        confidence = probability[0][prediction[0]]
        
        # Return the result as JSON
        return jsonify({
            'prediction': label,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)