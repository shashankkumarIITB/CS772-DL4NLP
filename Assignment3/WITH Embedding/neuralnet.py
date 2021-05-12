from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Embedding, InputLayer
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.initializers import Constant

def softmax_activation(x):
    # write your own implementation from scratch and return softmax values (using predefined softmax is prohibited)
    x_max = tf.math.reduce_max(x, axis=1, keepdims=True)
    x_exp = tf.math.exp(tf.math.subtract(x, x_max))
    x_exp_sum = tf.math.reduce_sum(x_exp, axis=1, keepdims=True)
    return tf.divide(x_exp, x_exp_sum)

class NeuralNet:
    # Class for defining the neural network architecture

    def __init__(self, reviews, ratings, vocab_size, embedding_dim, embedding_matrix, max_input_length=15, max_ratings=5, split_size=0.1, epochs=4, batch_size=32):
        # Maximum length of input sequences
        self.max_input_length = max_input_length
        # Maximum ratings for a review
        self.max_ratings = max_ratings
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        # Length of the vocabulary
        self.vocab_size = vocab_size
        # Embedding matrix
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        # Split dataset into training and validation sets
        self.reviews_train, self.reviews_validation, self.ratings_train, self.ratings_validation = train_test_split(reviews, ratings, test_size=split_size)
        # Print insights about the data
        self.print_insights(ratings)

    def print_insights(self, ratings):
        # Print information about the training data available
        ratings = [np.where(r==1)[0][0] for r in np.array(ratings)]
        ratings, count = np.unique(ratings, return_counts=True)
        for r, c in zip(ratings, count):
            print(f'Ratings: {r+1} => {c}') 

    def build_nn(self, bi, ci, di):
        #add the input and output layer here; you can use either tensorflow or pytorch
        model = Sequential()
        model.add(InputLayer(input_shape=(self.max_input_length, ), name='input'))
        model.add(Embedding(self.vocab_size, self.embedding_dim, embeddings_initializer=Constant(self.embedding_matrix), trainable=False, name='embedding'))
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