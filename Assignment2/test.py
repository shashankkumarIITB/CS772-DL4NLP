# Disable tensorflow logs
import os,csv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from neuralnet import NeuralNet
from preprocess import preprocess_data

# custom reviews to test the model
# reviews = [
#     'Very bad product. I would never recommend this to anyone. Please stop selling fake products. Utter waste of money.', 
#     "This book was very informative, covering all aspects of game.",
#     "I am already a baseball fan and knew a bit about the Negro leagues, but I learned a lot more reading this book.",
#     "Solid construction, good fit and finish.  The razor side just fits my razor.",
# ]

test_data = []
test_ratings = []
with open('gold_test.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        test_data.append(row['reviews'])
        ratings = int(row['ratings'])
        # Convert output ratings to one-hot encoding 
        test_ratings.append(list(np.eye(5)[ratings-1]))


# load the saved model
nn_relu = NeuralNet.load_nn()
# nn_sigmoid = NeuralNet.load_nn_sigmoid()
# preprocess the data
reviews_processed = preprocess_data(test_data)
# get the predictions from the model
predictions_softmax_relu = nn_relu.predict(reviews_processed)
# predictions_softmax_sigmoid = nn_sigmoid.predict(reviews_processed)
# convert the softmax predictions into class labels
predictions_relu = predictions_softmax_relu.argmax(axis=1) + 1
# predictions_sigmoid = predictions_softmax_sigmoid.argmax(axis=1) + 1

# test_file = open("test_predict_sigmoid.txt","w+")

# for p in predictions:
#     test_file.write(f'{prediction}\n')

# test_file.close()

test_size = len(predictions_relu)
match_count = 0

for i in range(test_size):
	p = predictions_relu[i]
	t = test_ratings[i]

	if t[p-1]==1.0:
		match_count += 1

relu_acc = match_count*100/test_size

# match_count = 0

# for i in range(test_size):
# 	p = predictions_sigmoid[i]
# 	t = test_ratings[i]

# 	if t[p-1]==1.0:
# 		match_count += 1

# sigmoid_acc = match_count*100/test_size

print('> test accuracy with relu+sigmoid activation = %.3f' % (relu_acc))
# print('> test accuracy with sigmoid activation = %.3f' % (sigmoid_acc))
# print the results
# for review, prediction in zip(reviews, predictions):
#     print(f'{review} => {prediction}')