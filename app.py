pip install flask tensorflow nltk numpy pandas


from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load trained LSTM model and tokenizer
model = tf.keras.models.load_model("optimized_lstm_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Set max length for padding (should match training)
max_length = 50

# Preprocessing function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Route for UI
@app.route('/')
def home():
    return render_template("index.html")

# API for predicting score
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['answer']
    
    # Preprocess and tokenize input
    cleaned_text = preprocess_text(data)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Predict score
    predicted_score = model.predict(padded_sequence)[0][0]
    
    # Clip score to valid range (0-5)
    predicted_score = np.clip(predicted_score, 0, 5)
    
    return jsonify({'predicted_score': round(predicted_score, 2)})

if __name__ == '__main__':
    app.run(debug=True)
