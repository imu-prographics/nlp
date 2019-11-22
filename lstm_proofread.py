%matplotlib inline

from __future__ import unicode_literals, print_function, division
from io import open
import string
import random
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

SOS_token = 0
EOS_token = 1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



class Data:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "[SOS]", 1: "[EOS]"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

MAX_LENGTH = 30

def prepareData(input_data_path, num_samples=10000):
    with  open(input_data_path, 'r', encoding='utf-8') as f:
        input_lines = f.read().split('\n')  # 行ごとのリストに
    data = Data()
    for line in input_lines:
        input_words = line.split(" ")
        target_words = input_words[1:]
        target_words.append("[EOS]")
        target_lines.append(" ".join(target_words))
        data.addSentence(line)
    min_samples = min(num_samples, min(len(input_lines)-1, len(target_lines)-1))
    pairs = [[i,t]for (i,t) in zip(input_lines[:min_samples],target_lines[:min_samples])]
    pairs.append(input_lines)
    pairs.append(target_lines)
    pairs = filterPairs(pairs)

    print("Counted words:")
    print('data:{}'.format(data.n_words))
    return data, pairs

input_data_path = "./data/text/processed/merged_normal.txt"
data, pairs = prepareData(input_data_path, 30000000)
print(len(pairs))
print(random.choice(pairs))

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.initHidden()

    def forward(self, input, hidden):
        embeds = self.embeds(input)
        lstm_out, hidden = self.lstm(
            embeds.view(len(input), 1, -1), hidden)
        output = self.linear(lstm_out.view(len(input), -1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

def indexesFromSentence(data, sentence):
    return [data.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(data, sentence):
    indexes = indexesFromSentence(data, sentence)
    #indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(encoder_data, pair[0])
    target_tensor = tensorFromSentence(decoder_data, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, model, optimizer, criterion, max_length=MAX_LENGTH):
    hidden = model.initHidden()
    optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    for i in range(input_length):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])
    loss.backward()

    optimizer.step()
    return loss.item() / target_length


def trainIters(model, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(model, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(data, sentence)
        input_length = input_tensor.size()[0]
        hidden = model.initHidden()
        outputs = torch.zeros(max_length, model.hidden_size, device=device)
        for i in range(input_length):
            output, hidden = model(input_tensor[i], hidden)
            outputs[i] = output

        return outputs

def evaluateRandomly(model, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        outputs = evaluate(model, pair[0])
        input_words = pair[0].split(" ")
        for j, output in enumerate(outputs):
            _, predict = output.data.topk(5)
            output_words =  [data.index2word(k) for k in predict]  
            print(input_words[j] +":"+ str(output_words))
        print('')

hidden_size = 256
model = LSTMPredictor(data.n_words, hidden_size).to(device)

trainIters(model, 1000, print_every=10)

evaluateRandomly(model)

outputs = evaluate(model, "だが 「 市場 の 論理 」 万能 の グローバル 化 は どう か 。")
