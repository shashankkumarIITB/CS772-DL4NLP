import csv, re
import numpy as np
from nltk.corpus import stopwords
import gensim
import numpy as np

# Maxlength of the sequence of input words
MAX_INPUT_LENGTH = 15
# Maximum output ratings
MAX_RATINGS = 5
# Dictionary and index for encoding
UNKNOWN_TOKEN = 'UKN'
WORD_TO_INDEX = {UNKNOWN_TOKEN: 0}
# Embeddings to be used
EMBEDDING_DIM = 300
EMBEDDING_MATRIX = np.array([np.zeros((EMBEDDING_DIM,))])
WORD2VEC_EMBEDDING = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
# Stopwords = Words that do not add meaning to the sentence
STOPWORDS = stopwords.words("english")

def get_train_data(train_file, max_ratings=MAX_RATINGS):
    # Split training reviews and ratings from the train file
    train_data = []
    train_ratings = []
    with open(train_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            train_data.append(row['reviews'])
            ratings = int(row['ratings'])
            # Convert output ratings to one-hot encoding 
            train_ratings.append(list(np.eye(max_ratings)[ratings-1]))
    return train_data, train_ratings

def get_test_data(test_file):
    # Return test reviews given test file
    test_data = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            test_data.append(row['reviews'])
    return test_data

def get_test_ratings(test_file):
    # Return test ratings given test file
    test_ratings = []
    with open(test_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            ratings = int(row['ratings'])
            # Convert output ratings to one-hot encoding 
            test_ratings.append(list(np.eye(5)[ratings-1]))
    return test_ratings

def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return text.lower()

def remove_punctuation(text):
    # return the reviews after removing punctuations
    punctuations = ['.', ',', ';', '"', "'", '?', '!', '-', '~', ':', '(', ')', '{', '}', '[', ']', '%', '_', '$', '#', '&', '^', '@']
    for e in punctuations:
        text = text.replace(e, ' ')
    return re.sub('[0-9\s]+', ' ', text)

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    return ' '.join([word for word in text.split() if word not in STOPWORDS])

def perform_tokenization(text):
    # return the reviews after performing tokenization
    return text.split()

def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    # return encoded examples
    global WORD_TO_INDEX, EMBEDDING_MATRIX
    vocab_size = len(WORD_TO_INDEX)
    encoding = {}
    for word in text:
        if word not in WORD_TO_INDEX:
            # Add the word to the dictionary
            WORD_TO_INDEX[word] = vocab_size
            # Add the embeddings for the word
            try:
                EMBEDDING_MATRIX = np.append(EMBEDDING_MATRIX, np.reshape(WORD2VEC_EMBEDDING[word], (1, -1)), axis=0)
            except KeyError:
                EMBEDDING_MATRIX = np.append(EMBEDDING_MATRIX, [EMBEDDING_MATRIX[0]], axis=0)
            vocab_size += 1
        encoding[word] = WORD_TO_INDEX[word]
    return encoding

def perform_padding(data, max_input_length=MAX_INPUT_LENGTH):
    # return the reviews after padding the reviews to maximum length
    # input data would be a dict of encoded review
    return [val for val in list(data.values())[:max_input_length]] + [0] * (max_input_length - len(data))

def preprocess_data(data, max_input_length=MAX_INPUT_LENGTH):
    # make all the following function calls on your data
    # return processed data
    # Data is a list of reviews
    processed_data = []
    for review in data:
        review = convert_to_lower(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = perform_tokenization(review)
        review = encode_data(review)
        review = perform_padding(review, max_input_length)
        processed_data.append(review)
    return processed_data
