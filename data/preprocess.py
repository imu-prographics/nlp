import sys 
import glob
import re
import MeCab
import os



def pick_article(read_paths):
    output_seq = []
    patterns = ["＼ＨＯＮ＼","＼Ｔ２＼"]
    for read_path in read_paths:
        with open(read_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                for i in range(len(patterns)):
                    m = re.match(patterns[i],line)
                    if m:
                        line = re.sub("＼ＨＯＮ＼","",line)
                        line = re.sub("＼Ｔ２＼","",line)
                        result_seq.append(line)
    return output_seq



def delete_noise(input_seq):
    output_seq = []
    patterns = ["【.*】","◆","★","■","□""◇","◇.*\n","＊","〇　…","○　…","…　…","（.*）",\
            "＜.*＞","——","▽.*\n","▼","＝.*＝","×××","●.*\n","☆","^[^　].*\n","×",\
            "ｈｔｔｐ.*／","^　[Ａ-Ｚ]　","^　一、","^[０-９]　.*\n","^.*[０-９]{5,}.*\n",\
            "^.{1,20}\n","^.*[０-９]{1,5}　{1,10}[０-９]{1,5}.*\n","^［.*\n",\
            "^.*[０-９]{3,4}・[０-９]{3,4}.*\n","[０-９]{2,4}・[０-９]{2,4}・[０-９]{2,4}",\
            "^　.{1,5}　","^　[０-９]月.[０-９]日　","^　◎","^　.*委員長　","Ｑ[０-９]．",\
            "》","《"]
        lines = of.readlines()
        for line in input_seq:
            for i in range(len(patterns)):
                line = re.sub(patterns[i],"",line)
                line = re.sub("　{2-30}","　",line)
                line = re.sub("▲","。",line)
            
            m = re.match("^　.*\n",line)
            if m:
                line = re.sub("^△","",line)
                for i in range(10):
                    line = re.sub("^.*・[０-９]{2,}、[０-９]{2,}年.*\n","",line)
                    line = re.sub("^.*　{2,}.*\n","",line)
                    line = re.sub("^　","",line)
                    line = re.sub("^　.*\n","",line)
                    line = re.sub("^\n","",line)
                output_seq.append(line)
    # 文のリストを返す
    return output_seq 

def separate_line(input_seq):
    output_seq = []
    flag_maru = False
    flag_kagi = False
    for line in input_seq:
            tokens = list(line)
            for i, token in enumerate(tokens):
                output_seq.append(token)
                if token == '「':
                    flag_kagi = True
                if token == '」':
                    flag_kagi = False
                if token == '（':
                    flag_maru = True
                if token == '）':
                    flag_maru = False
                
                if flag_kagi == False and flag_maru == False:
                    if token == '。':
                        output_seq.append(' \n')
                    if token == '？':
                        output_seq.append('\n')
    # 文のリストを返す
    return output_seq

def mecab_separate(input_seq):
    output_seq = []
    mt = MeCab.Tagger("-Ochasen")
    for line in input_seq:
            mt_lines = mt.parse(line).split("\n")
            pickup_array = []
            for mt_line in mt_lines:
                mt_array = mt_line.split()
                if len(mt_array) < 3:
                    continue
                pickup_array.append(mt_array[0])
                pickup_array.append(mt_array[3])
                line = "\t".join(pickup_array)+"\n"
                line = re.sub("-.*\n","\n",line) #単語と品詞以外削除
                output_seq.append(line)
                pickup_array = []
            output_seq.append("EOL\n")
        output_seq.append("EOF\n")
    # 文のリストを返す
    return output_seq

def delete_needless_lines(input_seq):
    output_seq = []
    flag = False
    line_stack = []
    for line in input_seq:
            if re.search("助詞|助動詞|動詞",line):
                flag = True
            line_stack.append(line)
            if re.match("EOL",line):
                if flag is True:
                    output_seq.append("\n".join(line_stack))
                    output_seq.append('\n')
                    flag = False
                line_stack = []
    # 文のリストを返す
    return output_seq

def create_class_masked_corpus(input_seq, mask_class, rate):
    output_seq = []
    line_stack = []
    for line in input_seq:
        word_list = line.split()
        if re.match("EOL", line):
            line_stack.append("\n")
            output_seq.append(line_stack)
            line_stack.clear()
        if len(word_list)==2:
            if re.match(mask_class, word_list[1]):
                if random.random() <= rate:
                    word_list[0] = "<MASK>"
            line_stack.append(word_list[0]+" ")
    # 文のリストを返す
    return output_seq

def create_word_masked_corpus(input_seq, mask_class, mask_word):
    output_seq = []
    line_stack = []
    for line in input_seq:
        word_list = line.split()
        if re.match("EOL", line):
            line_stack.append("\n")
            output_seq.append(line_stack)
            line_stack.clear()
        if len(word_list)==2:
            if re.match(mask_class, word_list[1]):
                if word_list[0] == mask_word:
                    word_list[0] = "<MASK>"
            line_stack.append(word_list[0]+" ")
    # 文のリストを返す
    return output_seq

def create_normal_corpus(input_seq):
    output_seq = []
    line_stack = []
    for line in input_seq:
            word_list = line.split()
            if re.match("EOL", line):
                line_stack.append("\n")
                output_seq.append(line_stack)
                line_stack.clear()
            if len(word_list)==2:
                line_stack.append(word_list[0]+" ")
    # 文のリストを返す
    return output_seq

def merge_data(input_seq, save_dir):
    with open(dir_name+"merged" + tag_name + ".txt","w",encoding="utf-8") as merged_file:
    for filename in files:
        with open(filename,"r",encoding="utf-8") as f:
            data = f.read()
            merged_file.write(data)


read_dir = "./text/original/"

read_paths = glob.glob(read_dir + "*.txt")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)