# Disable tensorflow logs
import sklearn
import os,csv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from neuralnet import NeuralNet
from preprocess import preprocess_data, get_test_data, get_test_ratings

# custom reviews to test the model
# reviews = [
#     'Very bad product. I would never recommend this to anyone. Please stop selling fake products. Utter waste of money.', 
#     "This book was very informative, covering all aspects of game.",
#     "I am already a baseball fan and knew a bit about the Negro leagues, but I learned a lot more reading this book.",
#     "Solid construction, good fit and finish.  The razor side just fits my razor.",
# ]

test_data = get_test_data('data/gold_test.csv')
test_ratings = get_test_ratings('data/gold_test.csv')

for b,c,d in l:
	print("bi = {}, ci = {}, di = {}".format(b,c,d))

	nn = NeuralNet.load_nn()
    # preprocess the data
    reviews_processed = preprocess_data(test_data)
    # get the predictions from the model
    predictions_softmax = nn.predict(reviews_processed)
    # convert the softmax predictions into class labels
    predictions = predictions_softmax.argmax(axis=1) + 1

    test_size = len(predictions)
    match_count = 0

    for i in range(test_size):
      if predictions[i] == test_ratings[i]:
        match_count += 1

    acc = match_count*100.0/test_size

    print('> test accuracy = %.3f' % (acc))

    print('\n')
    y_true = test_ratings
    y_pred = predictions

    print(sklearn.metrics.classification_report(y_true, y_pred))
    print(sklearn.metrics.confusion_matrix(y_true, y_pred))

    for i in range(2):
      print('\n')