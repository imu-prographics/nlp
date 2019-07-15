import sys 
import glob
import re
import os
import random
from argparse import ArgumentParser

def parser():
    usage = 'Usage: python -m mask_class -t tag_name'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('--mask', '-m', type=str, required=True, default="" , help='set class of mask word')
    argparser.add_argument('--savetag', '-s', type=str, required=True, help='set tag')
    argparser.add_argument('--rate', '-r', type=float, required=True, help='set rate of replacing target')
    args = argparser.parse_args()
    return args

args = parser()
mask_class = args.mask
save_tag = args.savetag
rate_replace = args.rate

read_dir = "./text/mecab_particle/"
save_dir = "./text/masked/"
read_paths = glob.glob(read_dir + "*_mecab_particle.txt")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#random.seed(0)
for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    save_path = re.sub("_mecab_particle.txt","",save_path) +"_masked_"+save_tag+".txt"
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    print(save_path)
    with open(save_path,"w",encoding="utf-8") as sf:
        for line in lines:
            word_list = line.split()
            if re.match("EOL", line):
                sf.write("\n")
            if len(word_list)==2:
                if re.match(mask_class, word_list[1]):
                    if random.random() <= rate_replace:
                       word_list[0] = "MASK"
                sf.write(word_list[0]+" ")
