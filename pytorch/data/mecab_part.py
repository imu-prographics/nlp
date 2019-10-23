import sys 
import glob
import re
import MeCab
import os

read_dir = "./text/indention/"
save_dir = "./text/mecab/"
read_paths = glob.glob(read_dir + "*_indention.txt")
mt = MeCab.Tagger("-Ochasen")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    save_path = re.sub("_indention.txt","",save_path)+"_mecab.txt"
    print(save_path)
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    with open(save_path,"w",encoding="utf-8") as sf:
        for line in lines:
            mt_lines = mt.parse(line).split("\n")
            pickup_array = []
            for mt_line in mt_lines:
                mt_array = mt_line.split()
                if len(mt_array) < 3:
                    continue
                pickup_array.append(mt_array[0])
                pickup_array.append(mt_array[3])
                sf.write("\t".join(pickup_array)+"\n")
                pickup_array = []
            sf.write("EOL\n")
        sf.write("EOF\n")



