# Disable tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set the following environment vairables
# $env:FLASK_APP = "server.py"
# $env:FLASK_DEBUG = 1
# To start the server, use - flask run

from flask import Flask
from flask import request, render_template
from tensorflow.keras.models import load_model
from preprocess import preprocess_data, MAX_RATINGS
from neuralnet import softmax_activation
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home(title='Assignment 2'):
    if request.method == 'POST':
        review = [request.form['review']]
        model = request.form['model']
        model_layers = request.form['model-layer']
        hidden_layers = request.form['hidden-layer']
        if model_layers==5:
            model_name = f'models/{model}_{model_layers}_{hidden_layers}.h5'
        else:
            model_name = f'models/{model}_{model}_{hidden_layers}.h5'
        model = load_model(model_name, custom_objects={'softmax_activation': softmax_activation})
        review_processed = preprocess_data(review)
        prediction_softmax = model.predict(review_processed)
        prediction = prediction_softmax.argmax(axis=1) + 1
        predictions = [{'ratings': i+1, 'probability': prediction_softmax[0][i]} for i in range(MAX_RATINGS)]
        return render_template('home.html', title=title, prediction=prediction[0], predictions=predictions)
    return render_template('home.html', title=title)