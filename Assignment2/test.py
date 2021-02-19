# Disable tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from neuralnet import NeuralNet
from preprocess import preprocess_data

# custom reviews to test the model
reviews = [
    'Very bad product. I would never recommend this to anyone. Please stop selling fake products. Utter waste of money.', 
    "This book was very informative, covering all aspects of game.",
    "I am already a baseball fan and knew a bit about the Negro leagues, but I learned a lot more reading this book.",
    "Solid construction, good fit and finish.  The razor side just fits my razor.",
]

# load the saved model
nn = NeuralNet.load_nn()
# preprocess the data
reviews_processed = preprocess_data(reviews)
# get the predictions from the model
predictions_softmax = nn.predict(reviews_processed)
# convert the softmax predictions into class labels
predictions = predictions_softmax.argmax(axis=1) + 1
# print the results
for review, prediction in zip(reviews, predictions):
    print(f'{review} => {prediction}')