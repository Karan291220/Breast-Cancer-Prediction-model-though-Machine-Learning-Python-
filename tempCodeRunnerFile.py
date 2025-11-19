from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('model.pkl', 'rb'))

# set expected number of features
expected_feature_length = model.n_features_in_  # OR set = 30 if known

# flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features = features.split(',')
    
    try:
        np_features = np.asarray(features, dtype=np.float32)
    except ValueError:
        return render_template('index.html', message=["Invalid input format. Please enter numeric values only."])
    
    if len(np_features) != expected_feature_length:
        return render_template('index.html', message=[f"Expected {expected_feature_length} features, but got {len(np_features)}."])
    
    pred = model.predict(np_features.reshape(1, -1))
    message = ['Cancerous' if pred[0] == 1 else 'Not Cancerous']
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
