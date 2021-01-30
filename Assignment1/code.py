# ADD THE LIBRARIES YOU'LL NEED
import csv, re
from nltk.corpus import stopwords
from tensorflow.keras import Input, Dense

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''

# Stopwords = Words that do not add meaning to the sentence
STOPWORDS = stopwords.words("english")

# Maxlength of the sequence of words
MAX_LENGTH = 10

# Dictionary and index for encoding
UNKNOWN_TOKEN = 'UKN'
ENCODED_DICT = {UNKNOWN_TOKEN: 0}
INDEX = 1

def get_train_data(train_file):
    # Split training reviews and ratings from the train file
    train_data = []
    train_ratings = []
    with open(train_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            train_data.append(row['reviews'])
            train_ratings.append(row['ratings'])
    return train_data, train_ratings

def get_test_data(test_file):
    # Return test reviews given test file
    test_data = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            test_data.append(row['reviews'])
    return test_data

def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    
    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like: 
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to 
    # encode the words into integers 

    # return encoded examples
    global ENCODED_DICT, INDEX
    encoding = {}
    for word in text:
        if word not in ENCODED_DICT:
            # Add the word to the dictionary
            ENCODED_DICT[word] = INDEX
            INDEX += 1
        encoding[word] = ENCODED_DICT[word]
    return encoding

def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return text.lower()

def remove_punctuation(text):
    # return the reviews after removing punctuations
    punctuations = ['.', ',', ';', '"', "'", '?', '!', '-']
    for e in punctuations:
        text = text.replace(e, '')
    return re.sub('\s+', ' ', text)

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    return ' '.join([word for word in text.split() if word not in STOPWORDS])

def perform_tokenization(text):
    # return the reviews after performing tokenization
    return text.split()

def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    # input data would be an encoded dictionary
    return [val for val in data.values()] + [0] * (MAX_LENGTH - len(data))

def preprocess_data(data):
    # make all the following function calls on your data
    # EXAMPLE:->
        # '''
        # review = data["reviews"]
        # review = convert_to_lower(review)
        # review = remove_punctuation(review)
        # review = remove_stopwords(review)
        # review = perform_tokenization(review)
        # review = encode_data(review)
        # review = perform_padding(review)
        # '''
    # return processed data
    # Data is a list of reviews
    processed_data = []
    for review in data:
        review = convert_to_lower(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = perform_tokenization(review)
        review = encode_data(review)
        review = perform_padding(review)
        processed_data.append(review)
    return processed_data    

def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    pass

class NeuralNet:

    def __init__(self, reviews, ratings):
        self.reviews = reviews
        self.ratings = ratings

    def build_nn(self):
        #add the input and output layer here; you can use either tensorflow or pytorch
        X = Input(shape=(MAX_LENGTH, ), name='input')
        X = Dense

    def train_nn(self,batch_size,epochs):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy
        pass

    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        pass

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs=1,1

    # Read data from the training file and split the input and output
    train_data, train_ratings = get_train_data(train_file)
    test_data = get_test_data(test_file)

    train_reviews=preprocess_data(train_data)
    test_reviews=preprocess_data(test_data)

    # model=NeuralNet(train_reviews,train_ratings)
    # model.build_nn()
    # model.train_nn(batch_size,epochs)

    # return model.predict(test_reviews)

# Main function to run code from command line
if __name__ == '__main__':
    main('train.csv', 'test.csv')