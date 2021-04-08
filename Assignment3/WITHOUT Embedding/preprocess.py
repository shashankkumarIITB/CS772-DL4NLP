import csv, re
import numpy as np
from nltk.corpus import stopwords
import gensim
import numpy as np

# Stopwords = Words that do not add meaning to the sentence
STOPWORDS = stopwords.words("english")

# Maxlength of the sequence of input words
MAX_INPUT_LENGTH = 15

# MAximum output ratings
MAX_RATINGS = 5

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
            ratings = int(row['ratings'])
            # Convert output ratings to one-hot encoding 
            train_ratings.append(list(np.eye(MAX_RATINGS)[ratings-1]))
    return train_data, train_ratings

def get_test_data(test_file):
    # Return test reviews given test file
    test_data = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            test_data.append(row['reviews'])
    return test_data

def get_test_ratings(test_file, max_ratings=MAX_RATINGS):
    # Return test ratings given test file
    test_ratings = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            ratings = int(row['ratings'])
            # Convert output ratings to one-hot encoding 
            test_ratings.append(list(np.eye(max_ratings)[ratings-1]))
    return test_ratings

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
    punctuations = ['.', ',', ';', '"', "'", '?', '!', '-', '~', ':', '(', ')', '{', '}', '[', ']', '%', '_']
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
    return [val for val in list(data.values())[:MAX_INPUT_LENGTH]] + [0] * (MAX_INPUT_LENGTH - len(data))

def preprocess_data(data):
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