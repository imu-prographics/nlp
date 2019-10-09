from __future__                 import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Dense, Activation
from keras.layers               import LSTM, GRU, SimpleRNN, Embedding
from keras.optimizers           import RMSprop, Adam
from keras.utils.data_utils     import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.noise         import GaussianDropout as GD
import numpy as np
import random
import sys
import tensorflow               as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import json
import MeCab

class LSTM_proofreading(self):
    def __init__(self):


    def parser():
        usage = 'Usage: python --file weight_file_name --mode [train or test]'
        argparser = ArgumentParser(usage=usage)
        argparser.add_argument('--mode','-m', dest='mode', type=str, choices=['train','test'])
        argparser.add_argument('--weights', '-w', type=str, required=True, help='set filename of weights for save or load')
        argparser.add_argument('--load_model', '-l', type=str, required=True, help='set filename of load_model for save or load')
        argparser.add_argument('--input', '-i', type=str, required=False, help='set filename of input data path')
        argparser.add_argument('--target', '-t', type=str, required=False, help='set filename of target data path')
        argparser.add_argument('--num_samples', '-s', type=int, default=1000, required=False, help='max samples for training')
        argparser.add_argument('--epochs', '-e', type=int, default=100, required=False, help='set number of epochs')
        args = argparser.parse_args()
        return args


    def build_model(maxlen=None, out_dim=None, in_dim=256):
        with tf.device("/cpu:0"):
            print('Build model...')
            model = Sequential()
            model.add(Embedding(input_dim=in_dim, output_dim=out_dim,mask_zero=True))
            model.add(LSTM(128*20, return_sequences=False))
            model.add(BN())
            model.add(Dense(out_dim))
            model.add(Activation('linear'))
            model.add(Activation('sigmoid'))
            #model.add(Activation('softmax'))
            optimizer = Adam()
            model.compile(loss='binary_crossentropy', optimizer=optimizer) 
        return model

    def create_dictionary(input_data_path,target_data_path, num_samples):
        input_texts = []
        target_texts = []
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
            input_texts.append(seq)
            
            for word in seq:
                if word not in dict_words:
                    dict_words.add(word)
            seq = target_text.split(' ')
            target_texts.append(seq)
        dict_words.add('MASK')
        dict_words = sorted(list(dict_words))
        num_words = len(dict_words)

        # 入力文と出力文それぞれで最大単語数計算
        max_input_seq_length = max([len(words) for words in input_texts])
        max_target_seq_length = max([len(words) for words in target_texts])

        print('Number of samples:', len(input_texts))
        print('Number of unique words:', num_words)
        print('Max sequence length for inputs:', max_input_seq_length)
        print('Max sequence length for outputs:', max_target_seq_length)
            # 単語にIDを割り振る
        word_index = dict(
            [(word, i) for i, word in enumerate(dict_words)])
        #文章を単語IDの配列で表す
        input_texts_id = list(map(lambda word: word_index(word) ,input_texts))
        target_texts_id = list(map(lambda word: word_index(word) ,target_texts))
        return word_index, input_texts_id, target_texts_id

    def split_train_val(data_list, rate):
        n_split = int(data_list[0]*rate)
        data_train, data_val = np.vsplit(np.array(data_list),[n_split]).tolist()
        return data_train, data_val


    def ppx(y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
        return perplexity

    def train(model, epochs, batch_size, input_train, input_val, target_train, target_val):
        row_train = input_train.shape[0]
        row_val = target_val.shape[0]
        n_batch = math.ceil(row_train/batch_size)
        loss_bk = 10000
        for j in range(0, epochs) :
            print("Epoch ", j+1, "/", epochs)
            loss_mean = 0
            k = 0
            for i in range(0, n_batch):
                start_train = i * batch_size
                end_train = min([(i+1)*batch_size, row_train])
                start_val = k*batch_size
                end_val = min([(k+1)*batch_size, row_val])
                input_train_batch = input_train[start_train:end_train,:]
                target_train_batch = target_train[start_train:end_train,:]
                input_val_batch = input_val[start_val:end_val,:]
                target_val_batch = target_val[start_val:end_val,:]
                if end_val%row_val==0:
                    k = 0
                else:
                    k += 1
                train_loss = model.train_on_batch([input_train_batch,target_train_batch])
                val_loss = model.test_on_batch([input_val_batch, target_val_batch])
                loss_mean += val_loss
                print("%d/%d train_loss:%f val_loss:%f" % (start_train, row_train, train_loss, val_loss))
            loss_mean = loss_mean / n_batch
            if j == 0 or loss_mean <= loss_bk:
                loss_bk = loss_mean
            else:
                print('EarlyStopping')
        
        json_string = model.to_json()
        open(os.path.join(dir_model, model_filename), 'w').write(json_string)
        model.save_weights(weights_filename)

def main():
    args = parser()
    input_data_path = args.input
    target_data_path = args.target
    num_samples = args.num_samples

    gpu_count = 2   
    epochs = 100  # Number of epochs to train for.
    batch_size = 100
    latent_dim = 256  # Latent dimensionality of the encoding space.

    word_index, input_texts_id, target_texts_id = create_dictionary(input_data_path, target_data_path)
    input_train, input_val = split_train_val(input_texts_id)
    target_train, target_val = split_train_val(target_texts_id)
    model = build_model()
    model = multi_gpu_model(model, gpus=gpu_count)
    model.compile(optimizer='rmsprop', loss=ppx) 

if __name__=="__main__":
    main()