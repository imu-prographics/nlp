import sys
args = sys.argv
read_file = args[1]
save_file = args[2]

min_len = 10

with open(read_file,"r",encoding="utf-8") as rf:
    lines = rf.read().split('\n')
sentences = []

with open(save_file,"w",encoding="utf-8") as sf:
    for line in lines:
        words = line.split(' ')
        sentences.append(words)
    max_words = max(len(words) for words in sentences)
    min_words = min(len(words) for words in sentences)
    mean = sum(len(words) for words in sentences)/len(sentences)
    print("単語数の平均は："+str(mean))
    print("単語数の最大数は："+str(max_words))
    print("単語数の最小数は："+str(min_words))
    for line in lines:
        words = line.split(' ')
        cut_words = words[0:int(mean+min_words/2)]
        if len(cut_words) > min_len:
            sf.write(" ".join(cut_words))
            sf.write(" \n")