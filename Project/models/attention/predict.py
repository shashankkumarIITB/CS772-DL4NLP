# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KdwjADVdZ_a1hifsHfiAQ2Vo9IQUbooZ
"""

import pandas as pd
import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
import string
from string import digits
import re
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, Concatenate, TimeDistributed
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import pickle as pkl
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

BASE_PATH = 'models/attention/'

"""attention.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1XrjPL3O_szhahYZW0z9yhCl9qvIcJJYW
"""


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape(
                                       (input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        # Be sure to call this at the end
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            # print('encoder_out_seq>', encoder_out_seq.shape)
            # print('decoder_out_seq>', decoder_out_seq.shape)
            pass

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(
                states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(
                K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                pass
                # print('wa.s>', W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                pass
                # print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(
                K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                pass
                # print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh,
                                  self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                pass
                # print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            # <= (batch_size, latent_dim
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        fake_state_c = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[-1])
        # <= (batch_size, enc_seq_len, latent_dim
        fake_state_e = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[1])

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


# loading the model architecture and asigning the weights
json_file = open(f'{BASE_PATH}NMT_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_loaded = model_from_json(loaded_model_json, custom_objects={
                               'AttentionLayer': AttentionLayer})
# load weights into new model
model_loaded.load_weights(f"{BASE_PATH}NMT_model_weight.h5")


with open(f'{BASE_PATH}NMT_Etokenizer.pkl', 'rb') as f:
    vocab_size_source, Eword2index, englishTokenizer = pkl.load(f)

with open(f'{BASE_PATH}NMT_Mtokenizer.pkl', 'rb') as f:
    vocab_size_target, Mword2index, marathiTokenizer = pkl.load(f)

# with open('NMT_data.pkl', 'rb') as f:
#     X_train, y_train, X_test, y_test = pkl.load(f)

Eindex2word = englishTokenizer.index_word
Mindex2word = marathiTokenizer.index_word

# model_loaded.summary()

latent_dim = 500
max_length_english = 20
# encoder inference
encoder_inputs = model_loaded.input[0]  # loading encoder_inputs
# loading encoder_outputs
encoder_outputs, state_h, state_c = model_loaded.layers[6].output

# print(encoder_outputs.shape)
# model_loaded.summary()

encoder_model = Model(inputs=encoder_inputs, outputs=[
                      encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,), name='input_x')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_y')
decoder_hidden_state_input = Input(shape=(max_length_english, latent_dim), name='input_z')

# Get the embeddings of the decoder sequence
decoder_inputs = model_loaded.layers[3].output

# print(decoder_inputs.shape)
dec_emb_layer = model_loaded.layers[5]

dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_lstm = model_loaded.layers[7]
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# attention inference
attn_layer = model_loaded.layers[8]
attn_out_inf, attn_states_inf = attn_layer(
    [decoder_hidden_state_input, decoder_outputs2])

concate = model_loaded.layers[9]
decoder_inf_concat = concate([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_dense = model_loaded.layers[10]
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,
                        decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = Mword2index['start']

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

            if(sampled_token != 'end'):
                decoded_sentence += ' '+sampled_token

                # Exit condition: either hit max length or find stop word.
                if (sampled_token == 'end' or len(decoded_sentence.split()) >= (26-1)):
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
        if((i != 0 and i != Mword2index['start']) and i != Mword2index['end']):
            newString = newString+Mindex2word[i]+' '
    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString+Eindex2word[i]+' '
    return newString


# print("some test Examples")
# for i in range(5):

#     print("Review:", seq2text(X_test[i]))
#     print("Original summary:", seq2summary(y_test[i]))
#     print("Predicted summary:", decode_sequence(X_test[i].reshape(1, 20)))
#     print("\n")


def predict(s):
    X = englishTokenizer.texts_to_sequences(s)
    # print(X)
    X = pad_sequences(X, maxlen=20, padding='post')
    # print(X)
    # print("Review:", seq2text(X[0]))
    # print("Original summary:",seq2summary(X))
    # print("Predicted summary:", decode_sequence(X.reshape(1, max_length_english)))
    return decode_sequence(X.reshape(1, max_length_english))


# s = "are you lost baby girl?"
# predict([s])
