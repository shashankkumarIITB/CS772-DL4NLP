# Disable tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3

from flask import Flask
from flask import request, render_template
from neuralnet import NeuralNet
from preprocess import preprocess_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home(title='Assignment 2'):
    if request.method == 'POST':
        review = request.form['review']
        nn = NeuralNet.load_nn()
        review_processed = preprocess_data(review)
        prediction_softmax = nn.predict(review_processed)
        prediction = prediction_softmax.argmax(axis=1) + 1
        return render_template('home.html', title=title, prediction=prediction)
    return render_template('home.html', title=title)