import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
import MeCab
from argparse import ArgumentParser
from keras.losses import categorical_crossentropy
from keras import backend as K
import math

import pdb; 

def parser():
    usage = 'Usage: python --file weight_file_name --mode [train or test]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('--mode','-m', dest='mode', type=str, choices=['train','test'])
    argparser.add_argument('--file', '-f', type=str, required=True, help='set filename of weights for save or load')
    args = argparser.parse_args()
    return args

def ppx(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return perplexity

args = parser()

weights_filename = args.file

gpu_count = 2
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
batch_size = 100
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 8000  # Number of samples to train on.
# Path to the data txt file on disk.

input_data_path = "./input.txt"
target_data_path = "./output.txt"
# 単語の配列で表された文の配列に変更
input_texts = []
target_texts = []

input_words = set()  # 追記
target_words = set()  # 追記
with  open(input_data_path, 'r', encoding='utf-8') as f:
    input_lines = f.read().split('\n')  # 行ごとのリストに

with open(target_data_path, 'r', encoding='utf-8') as f:
    target_lines = f.read().split('\n')

min_samples = min(num_samples, min(len(input_lines)-1, len(target_lines)-1))
for index, (input_line,target_line) in enumerate(zip(input_lines[:min_samples],target_lines[:min_samples])):
    input_text = input_line
    target_text = target_line
    # \tが開始記号で\nが終端記号とする
    target_text = '\t ' + target_text + ' \n'

    # 単語単位に分割
    words = []
    words = input_text.split(' ')
    input_texts.append(words)
    for word in words:
        if word not in input_words:
            input_words.add(word)
    words = target_text.split(' ')
    target_texts.append(words)
    for word in words:
        if word not in target_words:
            target_words.add(word)

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

# 入力文と出力文それぞれで最大単語数計算
max_encoder_seq_length = max([len(words) for words in input_texts])
max_decoder_seq_length = max([len(words) for words in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input words:', num_encoder_tokens)
print('Number of unique output words:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 単語にIDを割り振る
input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])

#input_textsは文の配列なので、単語の配列で構成された文の配列が必要？
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='uint8')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='uint8')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='uint8')

# 文のインデックスと単語のインデックスと単語を格納するデータを作成
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[word]] = 1
    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[word]] = 1
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1

# Define an input sequence and process it.
with tf.device("/cpu:0"):
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


if args.mode == 'train':
    n_split = int(encoder_input_data.shape[0]*0.8)
    encoder_train, encoder_val = np.vsplit(encoder_input_data,[n_split])   #エンコーダインプットデータを訓練用と評価用に分割
    decoder_train, decoder_val = np.vsplit(decoder_input_data,[n_split])   #デコーダインプットデータを訓練用と評価用に分割
    target_train, target_val = np.vsplit(decoder_target_data,[n_split])   #ラベルデータを訓練用と評価用に分割
    # Run training
    model = multi_gpu_model(model, gpus=gpu_count)
    model.compile(optimizer='rmsprop', loss=ppx) 
    row = encoder_train.shape[0]
    n_batch = math.ceil(row/batch_size)
    loss_bk = 10000
    for j in range(0, epochs) :
        print("Epoch ", j+1, "/", epochs)
        for i in range(0, n_batch):
            start = i * batch_size
            end = min([(i+1)*batch_size, row])
            encoder_train_batch = encoder_train[start:end,:]
            decoder_train_batch = decoder_train[start:end,:]
            target_train_batch = target_train[start:end,:]
            encoder_val_batch = encoder_val[start:end,:]
            decoder_val_batch = decoder_val[start:end,:]
            target_val_batch = target_val[start:end,:]
            train_loss = model.train_on_batch([encoder_train_batch,decoder_train_batch],target_train_batch)
            val_loss = model.test_on_batch([encoder_val_batch, decoder_val_batch] ,target_val_batch)
            print("%d/%d train_loss:%f val_loss:%f" % (start, row, train_loss, val_loss))
        if j == 0 or val_loss <= loss_bk:
            loss_bk = val_loss
        else:
            print('EarlyStopping')
            break
    # Save model
    model.save_weights(weights_filename)
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states
elif args.mode == 'test':
    model.load_weights(weights_filename)

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
# でコードの際にインデックスから単語を引くための逆引き辞書
reverse_input_token_index = dict(
    (i, word) for word, i in input_token_index.items())
reverse_target_token_index = dict(
    (i, word) for word, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_token_index[sampled_token_index]
        decoded_sentence += sampled_word
        if (sampled_word == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

if args.mode=='train':
    for seq_index in range(10):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)

        print('-')
        print('Input sentence:', ''.join(input_texts[seq_index]))
        print('Decoded sentence:', decoded_sentence)

elif args.mode=='test':
    print("会話テキストを入力してください\n")
    m = MeCab.Tagger("-Owakati")
    input_words = []
    input_seq = np.zeros((max_encoder_seq_length,num_encoder_tokens),dtype='float32')
    while True:
        input_text = input()  # 形態素解析前の平文 
        input_words = m.parse(input_text).split(' '|"　")  # 分かち書き
        for i,word in enumerate(input_words):
            input_seq = i, input_token_index[word] 
            # 文における単語のインデックス（登場順）と単語ID（既知語であれば）
            
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)