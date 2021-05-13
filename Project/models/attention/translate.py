import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Dropout
from tensorflow.python.keras.backend import dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def src_preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()
    # w = '<start> ' + w + ' <end>'
    return w


def trg_preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿,।])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)


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


def define_models(vocab_inp_size, vocab_tar_size):

    dims = 512

    # define training encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embed = Embedding(vocab_inp_size, dims)(encoder_inputs)
    
    # LSTM 1
    encoder_lstm1 = LSTM(dims, return_sequences=True, return_state=True)
    dropout1 = Dropout(0.2)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(dropout1(encoder_embed))

    # LSTM 2
    encoder_lstm2 = LSTM(dims, return_sequences=True, return_state=True)
    dropout2 = Dropout(0.2)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(dropout2(encoder_output1))
    
    encoder = LSTM(dims, return_state=True)
    dropout3 = Dropout(0.2)
    encoder_outputs, state_h, state_c = encoder(dropout3(encoder_output2))
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embeding_layer = Embedding(vocab_tar_size, dims)
    decoder_embed = decoder_embeding_layer(decoder_inputs)

    decoder_lstm = LSTM(dims, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(vocab_tar_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(dims,))
    decoder_state_input_c = Input(shape=(dims,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embed2 = decoder_embeding_layer(decoder_inputs)

    decoder_outputs2, state_h, state_c = decoder_lstm(
        decoder_embed2, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post', value=0, maxlen=20)
    return tensor, lang_tokenizer


def load_dataset(train_src, train_trg):
    df = pd.DataFrame(list(zip(train_src, train_trg)), columns =['English', 'Hindi'])
    
    df['English']=df['English'].apply(lambda x:src_preprocess_sentence(x).split(" "))
    df['Hindi']=df['Hindi'].apply(lambda x:trg_preprocess_sentence(x).split(" "))
    
    df['length_eng_sentence']=df['English'].apply(lambda x:len(x))
    df['length_hin_sentence']=df['Hindi'].apply(lambda x:len(x))

    df=df[df['length_eng_sentence']<=20]
    df=df[df['length_hin_sentence']<=20]
    
    en = df["English"].to_numpy()
    hi = df["Hindi"].to_numpy()
    
    input_tensor, train_src_tokenizer = tokenize(en)
    target_tensor, train_trg_tokenizer = tokenize(hi)
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    print(max_length_targ, max_length_inp)
    decoder_output_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor[:, 1:], maxlen=20, padding='post', value=0)
    return input_tensor, target_tensor, decoder_output_tensor, train_src_tokenizer, train_trg_tokenizer


def predict_sequence(infenc, infdec, source, n_steps, inp_lang, targ_lang):
    # encode
    source = src_preprocess_sentence(source).split(' ')
    source = inp_lang.texts_to_sequences([source])[0]
    tensor = np.array(source + [0] * (20 - len(source)))
    print(tensor)

    state = infenc.predict(tensor)
    print(state[0].shape)
    # start of sequence input
    target_seq = np.array(targ_lang.texts_to_sequences([['<start>']]))
    target_seq = tf.keras.preprocessing.sequence.pad_sequences(
        target_seq, maxlen=20, padding='post', value=0)[0]
    print(target_seq.shape)
    # collect predictions
    stop_cond = False
    decoded_sentence = ''

    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(np.argmax(yhat[0, -1, :], -1))
        # update state
        state = [h, c]
        # update target sequence
        target_seq = np.argmax(yhat[:, -1, :], -1)

        # print(targ_lang.sequences_to_texts([output]))
        if targ_lang.sequences_to_texts([[output[-1]]])[0] == '<end>':
            break

    print(targ_lang.sequences_to_texts([output]))
    return np.array(output)

# flat_list = list(np.concatenate(regular_list).flat)


train_src = pd.read_csv('../../Data/eng-hin/train.src', sep='\t', encoding='utf-8', header=None, na_filter=False)
train_trg = pd.read_csv('../../Data/eng-hin/train.trg', sep='\t', encoding='utf-8', header=None, na_filter=False)

train_src = np.array(train_src).reshape(train_src.shape[0])
train_trg = np.array(train_trg).reshape(train_trg.shape[0])

print('='*80)
print(' '*35+'DATA LOADING DONE')
print('='*80)


print(train_src[:5])
print(train_trg[:5])

input_tensor, decoder_input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(train_src, train_trg)

print('='*80)
print(' '*35+'PRE-PROCESS DONE')
print('='*80)


print(input_tensor[0], target_tensor[0], decoder_input_tensor[0])
print(len(input_tensor), len(target_tensor), len(decoder_input_tensor))

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

print('vocab size', vocab_inp_size, vocab_tar_size)

train, infenc, infdec = define_models(vocab_inp_size, vocab_tar_size)
# adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
train.compile(optimizer = 'rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy',  ignore_class_accuracy(0)])
train.summary()
# infenc.summary()
# infdec.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
train.fit([input_tensor, decoder_input_tensor], target_tensor, epochs=50, validation_split=0.01, batch_size=512, callbacks=[es])

train.save('../../Models/Encoder-Decoder/train.h5')
infenc.save('../../Models/Encoder-Decoder/infenc.h5')
infdec.save('../../Models/Encoder-Decoder/infdec.h5')


print(predict_sequence(infenc, infdec, 'Tom and i are friends.', 100, inp_lang, targ_lang))
print(predict_sequence(infenc, infdec, 'My wife is Chinese.', 100, inp_lang, targ_lang))