# Disable tensorflow logs
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

# test_file = open("test_predict_sigmoid.txt","w+")
# for p in predictions:
#     test_file.write(f'{prediction}\n')
# test_file.close()

test_size = len(predictions)
match_count = 0

for i in range(test_size):
	p = predictions[i]
	t = test_ratings[i]

	if t[p-1]==1.0:
		match_count += 1

acc = match_count*100.0/test_size

print('> test accuracy = %.3f' % (acc))

# print the results
# for review, prediction in zip(reviews, predictions):
#     print(f'{review} => {prediction}')