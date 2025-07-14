import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

with open('mnb.pkl', 'rb') as model_file:
    model_nb = pickle.load(model_file)
with open('TFIDFvector.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        new_review = [request.form['sentence']]  
        new_review_vectorized = tfidf_vectorizer.transform(new_review)
        
        probs = model_nb.predict_proba(new_review_vectorized)[0]
        predicted_class = np.argmax(probs)

        # Debugging: Print probabilities dan predicted class ke terminal
        print(f"Probabilities: {probs}")
        print(f"Predicted Class: {predicted_class}")

        # Pilih kelas dengan probabilitas tertinggi
        if probs[2] > 0.7:
            prediction_text = 'Neutral'
        elif probs[0] > probs[1]:  # Jika probabilitas negative lebih tinggi
            prediction_text = 'Negative'
        elif probs[1] > probs[0]:  # Jika probabilitas positive lebih tinggi
            prediction_text = 'Positive'
        else:
            prediction_text = "Tidak dapat menentukan sentimen"  # Jika semua probabilitas sama

        prob_negative = probs[0]
        prob_positive = probs[1]
        prob_neutral = probs[2]

        return render_template('index.html', 
                               prediction_text=f'Prediction: {prediction_text}', 
                               user_input=new_review[0],
                               prob_negative=prob_negative,
                               prob_positive=prob_positive,
                               prob_neutral=prob_neutral)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
