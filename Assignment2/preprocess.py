import csv, re
from nltk.corpus import stopwords
import numpy as np
import gensim

# Stopwords = Words that do not add meaning to the sentence
STOPWORDS = stopwords.words("english")
# Maxlength of the sequence of input words
MAX_INPUT_LENGTH = 15
# Maximum output ratings
MAX_RATINGS = 5
# Dictionary and index for encoding
UNKNOWN_TOKEN = 'UKN'
WORD_TO_INDEX = {UNKNOWN_TOKEN: 0}
INDEX = 1
# Embeddings
EMBEDDING_DIM = 300
WORD2VEC_EMBEDDINGS_MATRIX=None
WORD2VEC_EMBEDDINGS = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)

def get_embedding_matrix(vocab_size,vocab):
    # Word2Vec embeddings
    global WORD2VEC_EMBEDDINGS_MATRIX
    WORD2VEC_EMBEDDINGS_MATRIX= np.zeros((vocab_size,EMBEDDING_DIM))
    for word,i in vocab.items():
        try:
            WORD2VEC_EMBEDDINGS_MATRIX[i]=WORD2VEC_EMBEDDINGS[word]
        except KeyError:
            pass

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

def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    # return encoded examples
    global WORD_TO_INDEX, INDEX
    encoding = {}
    for word in text:
        if word not in WORD_TO_INDEX:
            # Add the word to the dictionary
            WORD_TO_INDEX[word] = INDEX
            INDEX += 1
        encoding[word] = WORD_TO_INDEX[word]
    return encoding

def get_word2vec_embeddings(data):
    # Word2Vec embeddings
    embeddings = []
    for word in list(data.keys()):
        try:
            embeddings.append(WORD2VEC_EMBEDDINGS[word])
        except KeyError:
            pass
    return embeddings

def perform_padding(data, max_input_length=MAX_INPUT_LENGTH):
    # return the reviews after padding the reviews to maximum length
    # input data would be a list of embeddings
    return [val for val in data[:max_input_length]] + [np.zeros((EMBEDDING_DIM,))] * (max_input_length - len(data))

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
        review = get_word2vec_embeddings(review)
        get_embedding_matrix(INDEX,WORD_TO_INDEX)
        review = perform_padding(review, max_input_length)
        processed_data.append(review)
    return processed_data    
