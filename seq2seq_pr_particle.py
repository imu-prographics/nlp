import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.models import model_from_json
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
    argparser.add_argument('--weights', '-w', type=str, required=True, help='set filename of weights for save or load')
    argparser.add_argument('--load_model', '-l', type=str, required=True, help='set filename of load_model for save or load')
    argparser.add_argument('--input', '-i', type=str, required=False, help='set filename of input data path')
    argparser.add_argument('--target', '-t', type=str, required=False, help='set filename of target data path')
    argparser.add_argument('--num_samples', '-s', type=int, default=1000, required=False, help='max samples for training')
    args = argparser.parse_args()
    return args

def split_train_val(data_list, rate):
        n_split = int(len(data_list)*rate)
        data_array = np.array([np.array(line) for line in data_list])
        data_train, data_val = np.vsplit(np.array(data_array),[n_split])
        return data_train, data_val

def ppx(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return perplexity

args = parser()

model_filename = args.load_model
weights_filename = args.weights

gpu_count = 2
epochs = 1  # Number of epochs to train for.
batch_size = 100
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = args.num_samples  # Number of samples to train on.
# Path to the data txt file on disk.

input_data_path = args.input
target_data_path = args.target
# 単語の配列で表された文の配列に変更
if args.mode == 'train':
    encoder_input_texts = []
    decoder_input_texts = []
    dict_words = set()  # 追記
    with  open(input_data_path, 'r', encoding='utf-8') as f:
        input_lines = f.read().split('\n')  # 行ごとのリストに

    with open(target_data_path, 'r', encoding='utf-8') as f:
        target_lines = f.read().split('\n')

    min_samples = min(num_samples, min(len(input_lines)-1, len(target_lines)-1))
    for index, (input_text,target_text) in enumerate(zip(input_lines[:min_samples],target_lines[:min_samples])):
        # \tが開始記号で\nが終端記号とする
        target_text = '\t ' + target_text + ' \n'
        # 単語単位に分割
        seq = []
        seq = input_text.split(' ')
        encoder_input_texts.append(seq)
        for word in seq:
            if word not in dict_words:
                dict_words.add(word)
        seq = target_text.split(' ')
        for word in seq:
            if word not in dict_words:
                dict_words.add(word)
        decoder_input_texts.append(seq)
    dict_words = sorted(list(dict_words))
    num_words = len(dict_words)

    # 入力文と出力文それぞれで最大単語数計算
    max_input_seq_length = max([len(seq) for seq in encoder_input_texts])
    max_target_seq_length = max([len(seq) for seq in decoder_input_texts])

    print('Number of samples:', len(encoder_input_texts))
    print('Number of unique words:', num_words)
    print('Max sequence length for inputs:', max_input_seq_length)
    print('Max sequence length for outputs:', max_target_seq_length)
        # 単語にIDを割り振る
    word_index = dict(
        [(word, i) for i, word in enumerate(dict_words)])
    
    #文章を単語IDの配列で表す
    encoder_input_data = np.zeros((len(encoder_input_texts), max_input_seq_length),dtype='uint8')
    decoder_input_data = np.zeros((len(encoder_input_texts), max_target_seq_length),dtype='uint8')
    decoder_target_data = np.zeros((len(encoder_input_texts), max_target_seq_length, len(word_index)),dtype='uint8')

    for i,(encoder_seq, decoder_seq) in enumerate(zip(encoder_input_texts, decoder_input_texts)):
        for j, (encoder_word, decoder_word) in enumerate(zip(encoder_seq, decoder_seq)):
            encoder_input_data[i,j] = word_index[encoder_word]
            decoder_input_data[i,j] = word_index[decoder_word] 
            if j > 0:
                decoder_target_data[i, j-1, word_index[decoder_word]] = 1 

    encoder_train, encoder_val = split_train_val(encoder_input_data, 0.8)
    decoder_train, decoder_val = split_train_val(decoder_input_data, 0.8)
    target_train, target_val = split_train_val(decoder_target_data, 0.8)
    
    # Define an input sequence and process it.

    with tf.device("/cpu:0"):
        encoder_inputs = Input(shape=(None,))
        encoder_emb = Embedding(num_words, latent_dim)
        x = encoder_emb(encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True)
        x , state_h, state_c = encoder(x)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, ))
        decoder_emb = Embedding(num_words, latent_dim)
        x = decoder_emb(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        x,_,_ = decoder_lstm(x, initial_state=encoder_states)
        decoder_dense = Dense(num_words, activation='softmax')
        decoder_outputs = decoder_dense(x)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model = multi_gpu_model(model, gpus=gpu_count)
    model.compile(optimizer='rmsprop', loss=ppx) 
    row_train = encoder_train.shape[0]
    row_val = encoder_val.shape[0]
    n_batch = math.ceil(row_train/batch_size)
    loss_bk = 10000
    k=0
    
    for j in range(0, epochs) :
        print("Epoch ", j+1, "/", epochs)
        loss_mean = 0
        for i in range(0, n_batch):
            start_train = i * batch_size
            end_train = min([(i+1)*batch_size, row_train])
            start_val = k*batch_size
            end_val = min([(k+1)*batch_size, row_val])
            encoder_train_batch = encoder_train[start_train:end_train,:]
            decoder_train_batch = decoder_train[start_train:end_train,:]
            target_train_batch = target_train[start_train:end_train,:]
            
            encoder_val_batch = encoder_val[start_val:end_val,:]
            decoder_val_batch = decoder_val[start_val:end_val,:]
            target_val_batch = target_val[start_val:end_val,:]
            
            if end_val%row_val==0:
                k = 0
            else:
                k += 1

            train_loss = model.train_on_batch([encoder_train_batch,decoder_train_batch],target_train_batch)
            val_loss = model.test_on_batch([encoder_val_batch, decoder_val_batch] ,target_val_batch)
            loss_mean += val_loss
            print("%d/%d train_loss:%f val_loss:%f" % (start_train, row_train, train_loss, val_loss))
        loss_mean = loss_mean / n_batch
        if j == 0 or loss_mean <= loss_bk:
            loss_bk = loss_mean
        else:
            print('EarlyStopping')
            break
    
    #json_string = model.to_json()
    #open(model_filename, 'w', encoding='utf-8').write(json_string)
    model.save_weights(weights_filename)
elif args.mode == 'test':
    json_string = open(model_filename,'r',encoding='utf-8').read()
    model = model_from_json(json_string)
    model.load_weights(weights_filename)
# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

x = decoder_emb(decoder_inputs)

decoder_outputs, state_h, state_c = decoder_lstm(x, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# デコードの際にインデックスから単語を引くための逆引き辞書
reverse_word_index = dict(
    (i, word) for word, i in word_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = word_index['\t']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_words, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a word
        sampled_word_index = np.argmax(output_words[0, -1, :])
        sampled_word = reverse_word_index[sampled_word_index]
        decoded_sentence += sampled_word
        if (sampled_word == '\n' or
            len(decoded_sentence) > max_target_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index

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
        print('Input sentence:', ''.join(encoder_input_texts[seq_index]))
        print('Decoded sentence:', decoded_sentence)

elif args.mode=='test':
    print("会話テキストを入力してください\n")
    m = MeCab.Tagger("-Owakati")
    input_words = []
    input_seq = np.zeros((max_encoder_seq_length, num_words),dtype='float32')
    while True:
        input_text = input()  # 形態素解析前の平文 
        input_words = m.parse(input_text).split(' '|"　")  # 分かち書き
        for i,word in enumerate(input_words):
            input_seq = i, input_word_index[word] 
            # 文における単語のインデックス（登場順）と単語ID（既知語であれば）
            
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', encoder_input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)