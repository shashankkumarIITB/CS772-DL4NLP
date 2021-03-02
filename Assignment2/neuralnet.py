from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, InputLayer
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall

def softmax_activation(x):
    # write your own implementation from scratch and return softmax values (using predefined softmax is prohibited)
    x_max = tf.math.reduce_max(x, axis=1, keepdims=True)
    x_exp = tf.math.exp(tf.math.subtract(x, x_max))
    x_exp_sum = tf.math.reduce_sum(x_exp, axis=1, keepdims=True)
    return tf.divide(x_exp, x_exp_sum)

class NeuralNet:
    # Class for defining the neural network architecture
    def __init__(self, reviews, ratings, word_to_index, max_input_length=15, max_ratings=5, split_size=0.1, epochs=4, batch_size=32):
        # Maximum length of input sequences
        self.max_input_length = max_input_length
        # Maximum ratings for a review
        self.max_ratings = max_ratings
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        # Mapping between words and index
        self.word_to_index = word_to_index
        # Dataset
        self.reviews = np.array(reviews)
        self.ratings = np.array(ratings)
        # Split into training and validation sets
        self.reviews_train, self.reviews_validation, self.ratings_train, self.ratings_validation = train_test_split(reviews, ratings, test_size=split_size)
        # Print insights about the data
        self.print_insights()

    def build_nn(self):
        #add the input and output layer here; you can use either tensorflow or pytorch
        model = Sequential()
        model.add(InputLayer(input_shape=(self.max_input_length, ), name='input'))
        model.add(Embedding(len(self.word_to_index), 64, input_length=self.max_input_length, name='embedding'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='hidden_1'))
        model.add(Dense(128, activation='relu', name='hidden_2'))
        model.add(Dense(self.max_ratings, name='dense'))
        model.add(Activation(softmax_activation, name='softmax'))
        self.model = model
        print(model.summary())

    def train_nn(self):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
        self.model.fit(x=self.reviews_train, y=self.ratings_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        self.model.evaluate(x=self.reviews_validation, y=self.ratings_validation)
        self.model.save('Assignment2_sigmoid.h5')
    
    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        return self.model.predict(reviews)

    def load_nn_relu():
        # function to load the neural network with relu activation used
        return load_model('Assignment2_relu.h5', custom_objects={'softmax_activation': softmax_activation})

    def load_nn_sigmoid():
        # function to load the neural network with sigmoid activation used
        return load_model('Assignment2_sigmoid.h5', custom_objects={'softmax_activation': softmax_activation})

    def print_insights(self):
        # Print information about the training data available
        ratings = [np.where(r==1)[0][0] for r in self.ratings]
        ratings, count = np.unique(ratings, return_counts=True)
        for r, c in zip(ratings, count):
            print(f'Ratings: {r+1} => {c}') 

    def write_file(self, filename='validation_output.txt'):
        # Function to write predicted values on the validation set to file
        predictions = self.predict(self.reviews_validation)
        with open(filename, 'w') as file:
            for prediction, actual in zip(predictions, self.ratings_validation):
                file.write(f'{prediction.argmax() + 1}, {actual.argmax() + 1}\n')
