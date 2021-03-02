# Disable tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ADD THE LIBRARIES YOU'LL NEED
from neuralnet import NeuralNet
from preprocess import WORD_TO_INDEX, preprocess_data, get_train_data, get_test_data

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file, load_model=False):
    # get the test data and preprocess it
    test_data = get_test_data(test_file)
    test_reviews = preprocess_data(test_data)
    if load_model:
        # load the pretrained model
        model = NeuralNet.load_nn()
        return model.predict(test_reviews)
    else:
        # Hyperparameters
        batch_size, epochs = 1, 3
        # Read data from the training file and split the input and output
        train_data, train_ratings = get_train_data(train_file)
        # Preprocess the training reviews
        train_reviews = preprocess_data(train_data)
        # build the model and train it
        nn = NeuralNet(train_reviews, train_ratings, WORD_TO_INDEX, epochs=epochs, batch_size=batch_size)
        nn.build_nn()
        nn.train_nn()
        # predict on test reviews
        return nn.predict(test_reviews)

# Main function to run code from command line
if __name__ == '__main__':
    predictions_softmax = main('train_balanced.csv', 'test.csv', load_model=False)
    predictions = predictions_softmax.argmax(axis=1) + 1

    # Write the predictions to a file
    with open('predict.txt', 'w') as file:
        for prediction in predictions:
            file.write(f'{prediction}\n')
