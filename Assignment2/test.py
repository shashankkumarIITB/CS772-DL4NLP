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

# load the saved model
nn = NeuralNet.load_nn()
# preprocess the data
reviews_processed = preprocess_data(test_data)
# get the predictions from the model
predictions_softmax = nn.predict(reviews_processed)
# convert the softmax predictions into class labels
predictions = predictions_softmax.argmax(axis=1) + 1

test_size = len(predictions)
match_count = 0

true_ratings = [r.index(1.0)+1 for r in list(test_ratings)]

for i in range(test_size):
	if predictions[i] == true_ratings[i]:
		match_count += 1

acc = match_count*100.0/test_size

print('> test accuracy = %.3f' % (acc))

# print('\n')
# for i in range(1,6):
#   print('{} has been predicted {} times'.format(i, list(predictions).count(i)))

# print('\n')
# for i in range(1,6):
#   print('{} has occurred {} times'.format(i, list(true_ratings).count(i)))

print('\n')
y_true = true_ratings
y_pred = predictions

print(sklearn.metrics.classification_report(y_true, y_pred))
print('\n')

print(sklearn.metrics.confusion_matrix(y_true, y_pred))