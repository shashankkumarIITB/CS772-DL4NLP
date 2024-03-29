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
# from preprocess import preprocess_data, MAX_RATINGS
# from neuralnet import softmax_activation
from nlp4 import predict  
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home(title='Assignment 2'):
    if request.method == 'POST':
        review = [request.form['review']]
        # model_name = f'models/5.h5'
        # model = load_model(model_name, custom_objects={'softmax_activation': softmax_activation})
        # review_processed = preprocess_data(review)
        # prediction_softmax = model.predict(review_processed)
        prediction_softmax = predict(review)
        prediction = prediction_softmax.argmax(axis=1) + 1
        predictions = [{'ratings': i+1, 'probability': prediction_softmax[0][i]} for i in range(MAX_RATINGS)]
        return render_template('home.html', title=title, prediction=prediction[0], predictions=predictions)
    return render_template('home.html', title=title)

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True, threaded=True,port=5000)