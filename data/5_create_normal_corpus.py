import sys 
import glob
import re
import os

read_dir = "./text/mecab_particle/"
save_dir = "./text/normal/"
read_paths = glob.glob(read_dir + "*.txt")

for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    with open(read_path, "r", encoding="utf-8") as of:
        print(save_path) 
        lines = of.readlines()
    with open(save_path, "w", encoding="utf-8") as sf:
        for line in lines:
            word_list = line.split()
            if re.match("EOL", line):
                sf.write("\n")
            if len(word_list)==2:
                sf.write(word_list[0]+" ")
