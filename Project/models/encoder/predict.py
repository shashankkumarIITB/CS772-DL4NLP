import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# BASE_PATH = '../../Models/Encoder-Decoder/'
BASE_PATH = 'models/encoder/'


with open(f'{BASE_PATH}NMT_Etokenizer_5.pkl', 'rb') as f:
    englishTokenizer = pkl.load(f)

with open(f'{BASE_PATH}NMT_Mtokenizer_5.pkl', 'rb') as f:
    hindiTokenizer = pkl.load(f)

Eindex2word = englishTokenizer.index_word
Mindex2word = hindiTokenizer.index_word
Eword2index = englishTokenizer.word_index
Mword2index = hindiTokenizer.word_index



def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.cast(y_true, 'int64')
        y_pred_class = K.argmax(y_pred, axis=-1)
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class),
                         'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy



dependencies = {
    'ignore_accuracy': ignore_class_accuracy(0)
}

encoder_model = tf.keras.models.load_model(
    f'{BASE_PATH}infenc_5.h5', custom_objects=dependencies)

# encoder_model.summary()

decoder_model = tf.keras.models.load_model(
    f'{BASE_PATH}infdec_5.h5', custom_objects=dependencies)

# decoder_model.summary()
max_length_english = 20

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    print(e_out)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = Mword2index['<start>']
    # print(e_out.shape, e_h.shape, e_c.shape, target_seq.shape)

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            break
        else:
            sampled_token = Mindex2word[sampled_token_index]

            if(sampled_token != '<end>'):
                decoded_sentence += ' '+sampled_token

                # Exit condition: either hit max length or find stop word.
                if (sampled_token == '<end>' or len(decoded_sentence.split()) >= (26-1)):
                    stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if((i != 0 and i != Mword2index['<start>']) and i != Mword2index['<end>']):
            newString = newString+Mindex2word[i]+' '
    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString+Eindex2word[i]+' '
    return newString



def predict(s):
    X = englishTokenizer.texts_to_sequences(s)
    X = pad_sequences(X, maxlen=20, padding='post')
    # print(X)
    # print("Review:", seq2text(X[0]))
    # print("Original summary:",seq2summary(X))
    # print("Predicted summary:", decode_sequence(X.reshape(1, max_length_english)))
    return decode_sequence(X.reshape(1, max_length_english))


# print(decoder_model.summary())
# s = "are you lost baby girl i am not  a girl"
# print(predict([s]))
