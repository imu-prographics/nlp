import sys 
import glob
import re
import os
import random
 
read_path = "./text/mecab_particle/mai2000a_mecab_particle.txt"
save_path_normal = "./text/mai2000a_normal.txt"
save_path_masked = "./text/mai2000a_masked_particle.txt"
rate_replace = 0.3
mask_class = '助詞'


with open(read_path, "r", encoding="utf-8") as of:
    lines = of.readlines()

with open(save_path_normal, "w", encoding="utf-8") as sfn, open(save_path_masked, "w", encoding="utf-8") as sfm:
    for line in lines:
        word_list = line.split()
        if re.match("EOL", line):
            sfn.write("\n")
            sfm.write("\n")
        if len(word_list)==2:
            if re.match(mask_class, word_list[1]):
                #if random.random() <= rate_replace:
                sfn.write(word_list[0]+" ")
                sfm.write("MASK ")
                continue
 
            sfm.write(word_list[0]+" ")
            sfn.write(word_list[0]+" ")

            
