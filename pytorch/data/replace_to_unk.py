import sys
import numpy as np
import re
args = sys.argv
read_file = args[1]
save_file = args[2]
flag_unk = False
with open(read_file,"r",encoding="utf-8") as rf:
    sentences = rf.read()

word_count_dict = {}
all_words = re.split('[ \n]', sentences)
for word in all_words:
    if word not in word_count_dict:
        word_count_dict[word] = 1
    else:
        word_count_dict[word] += 1


lines = sentences.split('\n')
with open(save_file,"w",encoding="utf-8") as sf:
    for line in lines:
        flag_unk = False
        words = line.split(' ')
        for i, word in enumerate(words):
            if word_count_dict[word] <= 30:
                flag_unk = True
                break
        if flag_unk is False:
            sf.write(' '.join(words))
            sf.write("\n")
