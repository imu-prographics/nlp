import numpy as np
import torch
import random

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, vocab, seq_len, label_path='None', encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line for line in f]
        if label_path:
            self.labels_data = torch.LongTensor(np.loadtxt(label_path))
        else:
            # ラベル不要の時はダミーデータを埋め込む
            self.labels_data = [0 for _ in range(len(self.datas))]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        seq = self.datas[item]
        seq_random, seq_label = self.random_word(seq)
        labels = self.labels_data[item]

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        seq = [self.vocab.sos_index] + seq_random + [self.vocab.eos_index]
        seq_label = [self.vocab.pad_index] + seq_label + [self.vocab.pad_index]
        bert_input = seq[:self.seq_len]
        bert_label = seq_label[:self.seq_len]
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "labels": labels}
        
        return {key: torch.tensor(value) for key, value in output.items()}
    
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        hiragana = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in hiragana]
        for i, token in enumerate(tokens):
            if self.is_train: # Trainingの時は確率的にMASKする
                prob = random.random()
            else:  # Predictionの時はMASKをしない
                prob = 1.0
            
            if self.vocab.stoi.get(token, self.vocab.unk_index) in hiragana:
                if prob < 0.3:
                    prob /= 0.3
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = self.vocab.mask_index
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.randrange(len(self.vocab))
                    # 10% randomly change token to current token
                    else:
                        tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    output_label.append(0)
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
        return tokens, output_label


class ReplaceDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, vocab, seq_len, label_path='None', encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line for line in f]
        if label_path:
            self.labels_data = torch.LongTensor(np.loadtxt(label_path))
        else:
            # ラベル不要の時はダミーデータを埋め込む
            self.labels_data = [0 for _ in range(len(self.datas))]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        seq = self.datas[item]
        seq_replace, seq_label, cls_label = self.replace_hiragana(seq)
        labels = self.labels_data[item]

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        seq = [self.vocab.sos_index] + seq_replace + [self.vocab.eos_index]
        seq_label = [self.vocab.pad_index] + seq_label + [self.vocab.pad_index]
        bert_input = seq[:self.seq_len]
        bert_label = seq_label[:self.seq_len]
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        output = {"bert_input": bert_input,
                  "token_label": bert_label,
                  "label": cls_label}
        
        return {key: torch.tensor(value) for key, value in output.items()}
    

    def replace_hiragana(self, sentence):
        tokens = sentence.split()
        output_label = []
        seq_label = 0
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        #hiragana = "がのをにへとでやはともかなねよぞわ"
        hiragana = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in hiragana]
        prob = random.random()
        if prob < 0.85:
            for i, token in enumerate(tokens):
                if self.is_train: # Trainingの時は確率的に
                    prob = random.random()
                else:
                    prob = 1.0
                # トークンが平仮名なら
                if self.vocab.stoi.get(token, self.vocab.unk_index) in hiragana:
                    # 15%の確率で置き換え
                    if prob < 0.15:
                        while True:
                            replace = random.choice(hiragana)
                            if tokens[i] != replace:
                                break
                        tokens[i] = replace
                        #output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(1)
                        seq_label = 1
                    else:
                        tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                        output_label.append(0)
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    output_label.append(0)
        else:
            for i, token in enumerate(tokens):
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
        return tokens, output_label, seq_label

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, vocab, seq_len, label_path='None', encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line for line in f]
        if label_path:
            self.labels_data = torch.LongTensor(np.loadtxt(label_path))
        else:
            # ラベル不要の時はダミーデータを埋め込む
            self.labels_data = [0 for _ in range(len(self.datas))]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        seq = self.datas[item]
        original = self.get_original(seq)
        #if self.is_train == True:
        seq_replace, seq_label, cls_label = self.multi_delete_add_replace(seq)
        #else:
        #    seq_replace, seq_label, cls_label = self.test_mlm(seq)
        labels = self.labels_data[item]
        seq = [self.vocab.sos_index] + seq_replace + [self.vocab.eos_index]
        seq_label = [self.vocab.pad_index] + seq_label + [self.vocab.pad_index]
        bert_input = seq[:self.seq_len]
        bert_label = seq_label[:self.seq_len]
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        if self.is_train:
            output = {"bert_input": bert_input,
                    "token_label": bert_label,
                    "label": cls_label}
        else:
            output = {"bert_input": bert_input,
                    "token_label": bert_label,
                    "label": cls_label,
                    "original":original}
        return {key: torch.tensor(value) for key, value in output.items()}

    def get_original(self, sentence):
        tokens = sentence.split()
        output_tokens = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens]
        return output_tokens

    def multi_delete_add_replace(self, sentence):
        error_flag = 0
        tokens = sentence.split()
        output_tokens = []
        output_label = []
        seq_label = 0
        joshi = "がのをにへとどでやしかはもばてで"
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        ka = "かきくけこ"
        sa = "さしすせそ"
        ta = "たちつてと"
        na = "なにぬねのん"
        ha = "はひふへほ"
        ma = "まみむめも"
        ya = "やゆよ"
        ra = "らりるれろ"
        wa = "わを"
        ga = "がぎぐげご"
        za = "ざじずぜぞ"
        da = "だぢづでど"
        ba = "ばびぶべぼ"
        alphabet = "ｋｓｔｎｈｍｙｒｗｇｚｄｂ"
        prob = random.random()
        if prob < 0.85:
            for i, token in enumerate(tokens):
                prob = random.random()
                #if seq_label == 0:
                # トークンが平仮名なら
                if token in hiragana:
                    # 衍字
                    if prob < 0.1 and error_flag <2:
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)
                        #衍字追加
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(6)
                        seq_label = 1
                        error_flag += 1
                    # 誤字
                    elif prob < 0.2 and error_flag <2:
                        prob_goji = random.random()
                        replace = token
                        if prob_goji < 0.5:
                            if token in ka:
                                replace = "ｋ"
                            elif token in sa:
                                replace = "ｓ"
                            elif token in ta:
                                replace = "ｔ"
                            elif token in na:
                                replace = "ｎ"
                            elif token in ha:
                                replace = "ｈ"
                            elif token in ma:
                                replace = "ｍ"
                            elif token in ya:
                                replace = "ｙ"
                            elif token in ra:
                                replace = "ｒ"
                            elif token in wa:
                                replace = "ｗ"
                            elif token in ga:
                                replace = "ｇ"
                            elif token in za:
                                replace = "ｚ"
                            elif token in da:
                                replace = "ｄ"
                            elif token in ba:
                                replace = "ｂ"
                        else:
                            if token in joshi:
                                while True:
                                    replace = random.choice(joshi)
                                    if token != replace:
                                        break
                            else:
                                while True:
                                    replace = random.choice(hiragana)
                                    if token != replace:
                                        break
                        output_tokens.append(self.vocab.stoi.get(replace, self.vocab.unk_index))
                        output_label.append(3)
                        seq_label = 1
                        error_flag += 1
                    
                    # 脱字
                    elif prob < 0.25 and  i > 0 and error_flag <2:
                        if output_label[-1] == 0:
                            output_label[-1] = 1
                        elif output_label[-1] == 3:
                            output_label[-1] = 4
                        elif output_label[-1] == 6:
                            output_label[-1] = 7
                        seq_label = 1
                        error_flag += 1
                    
                    elif prob < 0.3 and  i > 0 and error_flag <2:
                        #2moji
                        if output_label[-1] == 0:
                            output_label[-1] = 2
                        elif output_label[-1] == 3:
                            output_label[-1] = 5
                        elif output_label[-1] == 6:
                            output_label[-1] = 8
                        seq_label = 1
                        error_flag += 2
                    # 正
                    else:
                        if error_flag >= 2:
                            error_flag = 0
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)       
                    #else:
                    #    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    #    output_label.append(0)
                else:
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(0)
        else:
            for i, token in enumerate(tokens):
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(0)

        return output_tokens, output_label, seq_label

    def test(self, sentence):
        error_flag = 0
        tokens = sentence.split()
        output_tokens = []
        output_label = []
        seq_label = 0
        joshi = "がのをにへとどでやしかはもばてで"
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        ka = "かきくけこ"
        sa = "さしすせそ"
        ta = "たちつてと"
        na = "なにぬねのん"
        ha = "はひふへほ"
        ma = "まみむめも"
        ya = "やゆよ"
        ra = "らりるれろ"
        wa = "わを"
        ga = "がぎぐげご"
        za = "ざじずぜぞ"
        da = "だぢづでど"
        ba = "ばびぶべぼ"
        alphabet = "ｋｓｔｎｈｍｙｒｗｇｚｄｂ"
        prob = random.random()
        if prob < 0.5:
            for i, token in enumerate(tokens):
                # トークンが平仮名なら
                if token in hiragana and seq_label == 0:
                    # 衍字
                    prob = random.random()
                    if prob < 0.1 and error_flag <2:
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)
                        #衍字追加
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(6)
                        seq_label = 1
                        error_flag += 1
                    # 誤字
                    elif prob < 0.2 and error_flag <2:
                        prob_goji = random.random()
                        replace = token
                        if prob_goji < 0.5:
                            if token in ka:
                                replace = "ｋ"
                            elif token in sa:
                                replace = "ｓ"
                            elif token in ta:
                                replace = "ｔ"
                            elif token in na:
                                replace = "ｎ"
                            elif token in ha:
                                replace = "ｈ"
                            elif token in ma:
                                replace = "ｍ"
                            elif token in ya:
                                replace = "ｙ"
                            elif token in ra:
                                replace = "ｒ"
                            elif token in wa:
                                replace = "ｗ"
                            elif token in ga:
                                replace = "ｇ"
                            elif token in za:
                                replace = "ｚ"
                            elif token in da:
                                replace = "ｄ"
                            elif token in ba:
                                replace = "ｂ"
                        else:
                            if token in joshi:
                                while True:
                                    replace = random.choice(joshi)
                                    if token != replace:
                                        break
                            else:
                                while True:
                                    replace = random.choice(hiragana)
                                    if token != replace:
                                        break
                        output_tokens.append(self.vocab.stoi.get(replace, self.vocab.unk_index))
                        output_label.append(3)
                        seq_label = 1
                        error_flag += 1
                    
                    # 脱字
                    elif prob < 0.25 and  i > 0 and error_flag <2:
                        if output_label[-1] == 0:
                            output_label[-1] = 1
                        elif output_label[-1] == 3:
                            output_label[-1] = 4
                        elif output_label[-1] == 6:
                            output_label[-1] = 7
                        seq_label = 1
                        error_flag += 1
                    
                    elif prob < 0.3 and  i > 0 and error_flag <2:
                        #2moji
                        if output_label[-1] == 0:
                            output_label[-1] = 2
                        elif output_label[-1] == 3:
                            output_label[-1] = 5
                        elif output_label[-1] == 6:
                            output_label[-1] = 8
                        seq_label = 1
                        error_flag += 2
                    # 正
                    else:
                        if error_flag >= 2:
                            error_flag = 0
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)       
                    #else:
                    #    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    #    output_label.append(0)
                else:
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(0)
        else:
            for i, token in enumerate(tokens):
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(0)

        return output_tokens, output_label, seq_label
    
    
    def test_mlm(self, sentence):
        error_flag = 0
        seq_label = 1
        tokens = sentence.split()
        output_tokens = []
        output_label = []
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        for i, token in enumerate(tokens):
            # トークンが平仮名なら
            if token in hiragana:
                prob = random.random()
                # 誤字
                if prob < 0.1 and error_flag <2:
                    output_tokens.append(self.vocab.mask_index)
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    error_flag += 1 
                # 脱字
                elif prob < 0.15 and  i > 0 and error_flag <2:
                    output_tokens.append(self.vocab.mask_index)
                    error_flag += 1
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                
                elif prob < 0.2 and  i > 0 and error_flag <2:
                    #2moji
                    output_tokens.append(self.vocab.mask_index)
                    output_tokens.append(self.vocab.mask_index)
                    error_flag += 2
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                # 正
                else:
                    if error_flag >= 2:
                        error_flag = 0
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(0)       
            else:
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(0)
        return output_tokens, output_label, seq_label

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, vocab, seq_len, label_path='None', encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line for line in f]
        if label_path:
            self.labels_data = torch.LongTensor(np.loadtxt(label_path))
        else:
            # ラベル不要の時はダミーデータを埋め込む
            self.labels_data = [0 for _ in range(len(self.datas))]

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, item):
        seq = self.datas[item]
        original = self.get_original(seq)
        seq_replace, token_label, omission_label, cls_label = self.multi_label(seq)
        labels = self.labels_data[item]
        seq = [self.vocab.sos_index] + seq_replace + [self.vocab.eos_index]
        token_label = [self.vocab.pad_index] + token_label + [self.vocab.pad_index]
        omission_label = [self.vocab.pad_index] + omission_label + [self.vocab.pad_index]
        bert_input = seq[:self.seq_len]
        bert_label = token_label[:self.seq_len]
        omission_label = omission_label[:self.seq_len]
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), omission_label.extend(padding)
        if self.is_train:
            output = {"bert_input": bert_input,
                    "token_label": bert_label,
                    "omission_label": omission_label,
                    "label": cls_label}
        else:
            output = {"bert_input": bert_input,
                    "token_label": bert_label,
                    "label": cls_label,
                    "original":original}
        return {key: torch.tensor(value) for key, value in output.items()}

    def get_original(self, sentence):
        tokens = sentence.split()
        output_tokens = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens]
        return output_tokens

    def multi_label(self, sentence):
        deleted_num = 0
        error_flag = 0
        tokens = sentence.split()
        output_tokens = []
        output_label = []
        datsuji_label = []
        seq_label = 0
        joshi = "がのをにへとどでやしかはもばてで"
        hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわをん"
        ka = "かきくけこ"
        sa = "さしすせそ"
        ta = "たちつてと"
        na = "なにぬねのん"
        ha = "はひふへほ"
        ma = "まみむめも"
        ya = "やゆよ"
        ra = "らりるれろ"
        wa = "わを"
        ga = "がぎぐげご"
        za = "ざじずぜぞ"
        da = "だぢづでど"
        ba = "ばびぶべぼ"
        alphabet = "ｋｓｔｎｈｍｙｒｗｇｚｄｂ"
        prob = random.random()
        if prob < 0.85:
            for i, token in enumerate(tokens):
                prob = random.random()
                #if seq_label == 0:
                # トークンが平仮名なら
                if token in hiragana:
                    # 正字 = 0
                    #  誤字 = 1
                    #　衍字 = 2
                    #
                    # 衍字
                    if prob < 0.1 and error_flag <2:
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)
                        datsuji_label.append(0)
                        #衍字追加
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(2)
                        datsuji_label.append(0)
                        seq_label = 1
                        error_flag += 1
                    # 誤字
                    elif prob < 0.2 and error_flag <2:
                        prob_goji = random.random()
                        replace = token
                        if prob_goji < 0.5:
                            if token in ka:
                                replace = "ｋ"
                            elif token in sa:
                                replace = "ｓ"
                            elif token in ta:
                                replace = "ｔ"
                            elif token in na:
                                replace = "ｎ"
                            elif token in ha:
                                replace = "ｈ"
                            elif token in ma:
                                replace = "ｍ"
                            elif token in ya:
                                replace = "ｙ"
                            elif token in ra:
                                replace = "ｒ"
                            elif token in wa:
                                replace = "ｗ"
                            elif token in ga:
                                replace = "ｇ"
                            elif token in za:
                                replace = "ｚ"
                            elif token in da:
                                replace = "ｄ"
                            elif token in ba:
                                replace = "ｂ"
                        else:
                            if token in joshi:
                                while True:
                                    replace = random.choice(joshi)
                                    if token != replace:
                                        break
                            else:
                                while True:
                                    replace = random.choice(hiragana)
                                    if token != replace:
                                        break
                        output_tokens.append(self.vocab.stoi.get(replace, self.vocab.unk_index))
                        output_label.append(1)
                        datsuji_label.append(0)
                        seq_label = 1
                        error_flag += 1
                    
                    # 脱字
                    elif prob < 0.25 and  i > 0 and error_flag <2:
                        datsuji_label[-1] = 1
                        seq_label = 1
                        error_flag += 1
                    
                    elif prob < 0.3 and  i > 0 and error_flag <2:
                        #2moji
                        datsuji_label[-1] = 2
                        seq_label = 1
                        error_flag += 2
                    # 正
                    else:
                        if error_flag >= 2:
                            error_flag = 0
                        output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                        output_label.append(0)
                        datsuji_label.append(0)       
                    #else:
                    #    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    #    output_label.append(0)
                else:
                    output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(0)
                    datsuji_label.append(0)
        else:
            for i, token in enumerate(tokens):
                output_tokens.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(0)
                datsuji_label.append(0)

        return output_tokens, output_label, datsuji_label, seq_label