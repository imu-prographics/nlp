import sys 
import glob
import re
import MeCab
import os
import random


"""
記事の見出しと本文のみを抽出
"""
def pick_article(read_path):
    output_lines = []
    patterns = ["＼ＨＯＮ＼","＼Ｔ２＼"]
    with open(read_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            for i in range(len(patterns)):
                m = re.match(patterns[i],line)
                if m:
                    line = re.sub("＼ＨＯＮ＼","",line)
                    line = re.sub("＼Ｔ２＼","",line)
                    output_lines.append(line)
    return output_lines


"""
不要な記号や数字列を削除
"""
def delete_noise(input_lines):
    output_lines = []
    patterns = ["【.*】","◆","★","■","□""◇","◇.*\n","＊","〇　…","○　…","…　…","○…","…○","…","（.*）",\
            "＜.*＞","〈","〉","——","▽.*\n","▼","＝.*＝","×××","●.*\n","☆","^[^　].*\n","×",\
            "ｈｔｔｐ.*／","^　[Ａ-Ｚ]　","^　一、","^[０-９]　.*\n","^.*[０-９]{5,}.*\n",\
            "^.{1,20}\n","^.*[０-９]{1,5}　{1,10}[０-９]{1,5}.*\n","^［.*\n","［.{1,5}］",\
            "^.*[０-９]{3,4}・[０-９]{3,4}.*\n","[０-９]{2,4}・[０-９]{2,4}・[０-９]{2,4}",\
            "^　.{1,5}　","^　[０-９]月.[０-９]日　","^　◎","^　.*委員長　","Ｑ[０-９]．",\
            "》","《","＝写真","※","○","□"]
    for line in input_lines:
        for i in range(len(patterns)):
            line = re.sub(patterns[i], "", line)
            line = re.sub("　{2-30}", "　", line)
            line = re.sub("▲", "。", line)
        
        m = re.match("^　.*\n",line)
        if m:
            line = re.sub("^△","",line)
            line = re.sub("『", "「", line)
            line = re.sub("』", "」", line)
            for i in range(10):
                line = re.sub("^.*・[０-９]{2,}、[０-９]{2,}年.*\n","",line)
                line = re.sub("^.*　{2,}.*\n","",line)
                line = re.sub("^　","",line)
                line = re.sub("^　.*\n","",line)
                line = re.sub("^\n","",line)
            output_lines.append(line)
    # 文のリストを返す
    return output_lines 

"""
文章を「。」や「？」で区切る
括弧が完結していない場合は区切らない
"""
def separate_lines(input_lines):
    output_lines = []
    line_stack = []
    flag_maru = False
    flag_kagi = False
    for line in input_lines:
        tokens = list(line)
        for token in tokens:
            line_stack.append(token)
            if token == '「':
                flag_kagi = True
            if token == '」':
                flag_kagi = False
            if token == '（':
                flag_maru = True
            if token == '）':
                flag_maru = False
            
            if flag_kagi is False and flag_maru is False:
                if token == '。':
                    line_stack.append('\n')
                    output_lines.append("".join(line_stack))
                    line_stack = []
                if token == '？':
                    line_stack.append('\n')
                    output_lines.append("".join(line_stack))
                    line_stack = []
        output_lines.append("".join(line_stack))
        line_stack = []
    # 文のリストを返す
    return output_lines

"""
MeCabで単語と品詞情報取得
文の情報を保持するためにEOL挿入
"""
def mecab_separate(input_lines):
    output_lines = []
    mt = MeCab.Tagger("-Ochasen")
    for line in input_lines:
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
            output_lines.append(line)
            pickup_array = []
        output_lines.append("EOL\n")
    output_lines.append("EOF\n")
    # 文のリストを返す
    return output_lines

def delete_needless_lines(input_lines):
    output_lines = []
    flag = False
    line_stack = []
    for line in input_lines:
            if re.search("助詞|助動詞|動詞",line):
                flag = True
            line_stack.append(line)
            if re.match("EOL",line):
                if flag is True:
                    for line in line_stack:
                        output_lines.append(line)
                    flag = False
                line_stack = []
    # 文のリストを返す
    return output_lines

def create_class_masked_lines(input_lines, mask_class, rate):
    output_lines = []
    line_stack = []
    for line in input_lines:
        word_list = line.split()
        if re.match("EOL", line):
            line_stack.append("\n")
            output_lines.append(" ".join(line_stack))
            line_stack.clear()
        if len(word_list)==2:
            if re.match(mask_class, word_list[1]):
                if random.random() <= rate:
                    word_list[0] = "<MASK>"
            line_stack.append(word_list[0]+" ")
    # 文のリストを返す
    return output_lines

def create_word_masked_lines(input_lines, mask_class, mask_word):
    output_lines = []
    line_stack = []
    for line in input_lines:
        word_list = line.split()
        if re.match("EOL", line):
            line_stack.append("\n")
            output_lines.append(" ".join(line_stack))
            line_stack.clear()
        if len(word_list)==2:
            if re.match(mask_class, word_list[1]):
                if word_list[0] == mask_word:
                    word_list[0] = "<MASK>"
            line_stack.append(word_list[0]+" ")
    # 文のリストを返す
    return output_lines

def create_normal_lines(input_lines):
    output_lines = []
    line_stack = []
    for line in input_lines:
        word_info = line.split("\t")
        if re.match("EOL", line):
            line_stack.append("\n")
            output_lines.append(" ".join(line_stack))
            line_stack.clear()
        if len(word_info) == 2:
            line_stack.append(word_info[0]+" ")
    # 文のリストを返す
    return output_lines

def merge_data(lines_list, save_dir, tag_name):
    with open(save_dir + "merged_" + tag_name + ".txt","w",encoding="utf-8") as merged_file:
        for lines in lines_list:
            for line in lines:
                merged_file.write(line)

def get_lines_min_length(input_lines, min_len):
    output_lines = []
    for line in input_lines:
        words = line.split(' ')
        if len(words) >= min_len:
            output_lines.append(" ".join(words) + "\n")
    return output_lines

def get_lines_max_length(input_lines, max_len):
    output_lines = []
    lines_stack = []
    for line in input_lines:
        words = line.split(' ')
        if len(words) <= max_len:
            output_lines.append(" ".join(words) + "\n")
    return output_lines

def get_lines_max_min_length(input_lines, max_len, min_len):
    output_lines = []
    lines_stack = []
    for line in input_lines:
        words = line.split(' ')
        if len(words) <= max_len and len(words) >= min_len:
            output_lines.append(" ".join(words) + "\n")
    return output_lines

def replace_to_unk(input_lines, min_tf):
    output_lines = []
    word_count_dict = {}
    all_words = re.split('[ \n]', input_lines)
    for word in all_words:
        if word not in word_count_dict:
            word_count_dict[word] = 1
        else:
            word_count_dict[word] += 1
    for line in input_lines:
        words = line.split(' ')
        for i, word in enumerate(words):
            if word_count_dict[word] <= min_tf:
                words[i] = "<UNK>"
        output_lines.append(" ".join(words))
    return output_lines

def create_temp_files(read_dir, temp_dir):
    read_paths = glob.glob(read_dir + "*.txt")
    for read_path in read_paths:
        print("processing " + os.path.basename(read_path))
        print("本文抽出中")
        lines = pick_article(read_path)
        print("ノイズ削除中")
        lines = delete_noise(lines)
        print("文の分割中")
        lines = separate_lines(lines)
        print("単語分割中")
        lines = mecab_separate(lines)
        print("不要文削除中")
        lines = delete_needless_lines(lines)
        temp_path = temp_dir + os.path.basename(read_path)
        with open(temp_path,"w",encoding="utf-8") as sf:   
            for line in lines:
                sf.write(line)


"""
ここから一つのファイルを作成するメソッド
"""
def create_normal_corpus(temp_dir, save_dir):
    lines_list = []
    temp_paths = glob.glob(temp_dir + "*.txt")
    for i, temp_path in enumerate(temp_paths):
        with open(temp_path,"r",encoding="utf-8") as tf:
            lines = tf.readlines()
            print("コーパス作成中:{}/{}", i, len(temp_path))
            lines = create_normal_lines(lines)
            lines_list.append(lines)
    merge_data(lines_list, save_dir, "normal")

def create_class_masked_corpus(temp_dir, save_dir, mask_class, mask_rate):
    lines_list = []
    temp_paths = glob.glob(temp_dir + "*.txt")
    for i, temp_path in enumerate(temp_paths):
        with open(temp_path,"r",encoding="utf-8") as tf:
            lines = tf.readlines()
            print("コーパス作成中:{}/{}", i, len(temp_path))
            lines = create_class_masked_lines(lines, mask_class, mask_rate)
            lines_list.append(lines)
    merge_data(lines_list, save_dir, "particle")

def create_word_masked_corpus(temp_dir, save_dir, mask_class, mask_word):
    lines_list = []
    temp_paths = glob.glob(temp_dir + "*.txt")
    for i, temp_path in enumerate(temp_paths):
        with open(temp_path,"r",encoding="utf-8") as tf:
            lines = tf.readlines()
            print("コーパス作成中:{}/{}", i, len(temp_path))
            lines = create_word_masked_lines(lines, mask_class, mask_word)
            lines_list.append(lines)
    merge_data(lines_list, save_dir, "word")

def create_single_year_corpus():
    read_path = "./text/original/mai2000a.txt"
    save_path_normal = "./text/mai2000a_normal.txt"
    save_path_class_masked = "./text/mai2000a_particle.txt"
    save_path_word_masked = "./text/mai2000a_word.txt"
    mask_rate = 0.15
    mask_class = '助詞'
    mask_word = 'て'
    with open(read_path, "r", encoding="utf-8") as rf:
        lines = rf.readlines()
        print("processing " + os.path.basename(read_path))
        print("本文抽出中")
        lines = pick_article(read_path)
        print("ノイズ削除中")
        lines = delete_noise(lines)
        print("文の分割中")
        lines = separate_lines(lines)
        print("単語分割中")
        lines = mecab_separate(lines)
        print("不要文削除中")
        lines = delete_needless_lines(lines)
        print("コーパス作成中")
        lines_normal = create_normal_lines(lines)
        lines_class_masked = create_class_masked_lines(lines, mask_class, mask_rate)
        lines_word_masked = create_word_masked_lines(lines, mask_class, mask_word)
    
    with open(save_path_normal, "w", encoding='utf-8') as sf:
        for line in lines_normal:
            sf.write(line)
    
    with open(save_path_class_masked, "w", encoding='utf-8') as sf:
        for line in lines_class_masked:
            sf.write(line)
    
    with open(save_path_word_masked, "w", encoding='utf-8') as sf:
        for line in lines_word_masked:
            sf.write(line)

def delete_space(read_path, save_path):
    with open(read_path, "r", encoding="utf-8") as rf:
        lines = rf.readlines()
    with open(save_path, "w", encoding="utf-8") as sf:
        for line in lines:
            words = line.split()
            sf.write("".join(words)+"\n")


if __name__ == "__main__":
    read_dir = "./text/original/"
    temp_dir = "./text/temp/"
    save_dir = "./text/processed/"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    delete_space("./text/mai2000a_normal.txt", "./text/mai2000a_token.txt")
