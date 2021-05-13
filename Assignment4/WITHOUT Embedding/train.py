# Disable tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from neuralnet import NeuralNet
from preprocess import MAX_INPUT_LENGTH, preprocess_data, get_train_data, get_test_data, get_test_ratings

l = [[0,5,0], [0,5,1], [1,5,0], [1,5,1], [2,5,0], [2,5,1], [3,5,0], [3,5,1], [4,5,0], [4,5,1], 
     [0,0,0], [0,0,1], [1,1,0], [1,1,1], [2,2,0], [2,2,1], [3,3,0], [3,3,1], [4,4,0], [4,4,1]] # bi,ci,di

def main(train_file, test_file, load_model=False):
    # get the test data and preprocess it
    # test_data = get_test_data(test_file)
    # uncomment if test ratings are also available
    test_reviews = get_test_data(test_file)
    test_ratings = get_test_ratings(test_file)
    test_reviews = preprocess_data(test_reviews)
    if load_model:
        # load the pretrained model
        model = NeuralNet.load_nn()
        model.evaluate(test_reviews, test_ratings)
        return model.predict(test_reviews)
    else:
        # Hyperparameters
        batch_size, epochs = 256, 5
        # Read data from the training file and split the input and output
        train_data, train_ratings = get_train_data(train_file)
        # Preprocess the training reviews
        train_reviews = preprocess_data(train_data)
        # build the model and train it
        # for b,c,d in l:
        #     print("bi = {}, ci = {}, di = {}".format(b,c,d))

        #     nn = NeuralNet(train_reviews, train_ratings, len(WORD_TO_INDEX), EMBEDDING_DIM, EMBEDDING_MATRIX, epochs=epochs, batch_size=batch_size)
        #     nn.build_nn(b,c,d)
        #     nn.train_nn(bi=b, ci=c, di=d)

        #     for i in range(2):
        #         print('\n')
        nn = NeuralNet(train_reviews, train_ratings, epochs=epochs, batch_size=batch_size)
        b,c,d = 0,5,0
        nn.build_nn(b,c,d)
        nn.train_nn(bi=b, ci=c, di=d)
        # predict on test reviews
        return nn.predict(test_reviews)

# Main function to run code from command line
if __name__ == '__main__':
    predictions_softmax = main('data/train_balanced.csv', 'data/gold_test.csv', load_model=False)
    predictions = predictions_softmax.argmax(axis=1) + 1

    # # Write the predictions to a file
    # with open('predict.txt', 'w') as file:
    #     for prediction in predictions:
    #         file.write(f'{prediction}\n')
