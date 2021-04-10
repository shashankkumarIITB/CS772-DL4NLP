from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Embedding, InputLayer
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.initializers import Constant
from preprocess import MAX_INPUT_LENGTH, ENCODED_DICT, MAX_RATINGS

def softmax_activation(x):
    # write your own implementation from scratch and return softmax values (using predefined softmax is prohibited)
    x_max = tf.math.reduce_max(x, axis=1, keepdims=True)
    x_exp = tf.math.exp(tf.math.subtract(x, x_max))
    x_exp_sum = tf.math.reduce_sum(x_exp, axis=1, keepdims=True)
    return tf.divide(x_exp, x_exp_sum)

class NeuralNet:
    # Class for defining the neural network architecture

    def __init__(self, reviews, ratings, epochs, batch_size):
        self.reviews = np.array(reviews)
        self.ratings = np.array(ratings)
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        # Split into training and validation sets
        fraction_validation = 0.1
        length_validation = int(len(reviews) * fraction_validation)
        # Train set
        self.reviews_train = self.reviews[length_validation:]
        self.ratings_train = self.ratings[length_validation:]
        # Validation set
        self.reviews_validation = self.reviews[:length_validation]
        self.ratings_validation = self.ratings[:length_validation]
        # Print insights about the data
        self.print_insights()

    def print_insights(self):
        # Print information about the training data available
        ratings = [np.where(r==1)[0][0] for r in self.ratings]
        ratings, count = np.unique(ratings, return_counts=True)
        for r, c in zip(ratings, count):
            print(f'Ratings: {r+1} => {c}') 

    def build_nn(self, bi, ci, di):
        #add the input and output layer here; you can use either tensorflow or pytorch
        model = Sequential()
        model.add(InputLayer(input_shape=(MAX_INPUT_LENGTH, ), name='input'))
        model.add(Embedding(len(ENCODED_DICT), 300, name='embedding'))
        #model layers
        if bi==0:
        	model.add(LSTM(256, activation='sigmoid', name=f'LSTM_1', return_sequences=True))
        elif bi==1:
        	model.add(GRU(256, activation='sigmoid', name=f'GRU_1', return_sequences=True))
        elif bi==2:
        	model.add(SimpleRNN(256, activation='sigmoid', name=f'RNN_1', return_sequences=True))
        elif bi==3:
        	model.add(Bidirectional(LSTM(256, activation='sigmoid', return_sequences=True), name=f'bi-LSTM_1'))
        elif bi==4:
        	model.add(Bidirectional(GRU(256, activation='sigmoid', return_sequences=True), name=f'bi-GRU_1'))
        model.add(Dropout(0.2, name=f'dropout_1'))

        if ci==0:
        	model.add(LSTM(128, activation='sigmoid', name=f'LSTM_2', return_sequences=True))
        elif ci==1:
        	model.add(GRU(128, activation='sigmoid', name=f'GRU_2', return_sequences=True))
        elif ci==2:
        	model.add(SimpleRNN(128, activation='sigmoid', name=f'RNN_2', return_sequences=True))
        elif ci==3:
        	model.add(Bidirectional(LSTM(128, activation='sigmoid', return_sequences=True), name=f'bi-LSTM_2'))
        elif ci==4:
        	model.add(Bidirectional(GRU(128, activation='sigmoid', return_sequences=True), name=f'bi-GRU_2'))
        if ci in [0, 1, 2, 3, 4]:
        	model.add(Dropout(0.2, name=f'dropout_2'))

        model.add(Flatten(name='flatten'))
        if di == 1: #dense (hidden layer)
        	model.add(Dense(64, name='dense_0'))
        model.add(Dense(MAX_RATINGS, name='dense_1'))
        model.add(Activation(softmax_activation, name='softmax'))
        self.model = model
        print(model.summary())

    def train_nn(self, bi, ci, di):
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
        self.model.fit(x=self.reviews_train, y=self.ratings_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.reviews_validation, self.ratings_validation))
        self.model.evaluate(x=self.reviews_validation, y=self.ratings_validation)
        self.model.save(f'models/{bi}-{ci}-{di}.h5')
    
    def evaluate(self, reviews, ratings):
        return self.model.evaluate(reviews, ratings, self.batch_size)

    def load_nn(bi, ci, di):
        # function to load the neural network with relu activation used
        return load_model(f'models/{bi}-{ci}-{di}.h5', custom_objects={'softmax_activation': softmax_activation})

    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        return self.model.predict(reviews)