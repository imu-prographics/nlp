import sys 
import glob
import re
from argparse import ArgumentParser

def parser():
    usage = 'Usage: python --file weight_file_name --mode [train or test]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('--holdclass', '-c', type=str, required=False, default="" , help='set class of hold word')
    argparser.add_argument('--file', '-f', type=str, required=True, help='set save file name')
    args = argparser.parse_args()
    return args

args = parser()
hold_words_class = args.holdclass
save_file_tag = args.file

dir_name = "./text/"
open_file_names = glob.glob(dir_name+"*.txt")

for open_file_name in open_file_names:
    save_file_name = re.sub(".txt","",open_file_name)+save_file_tag
    with open(open_file_name, "r", encoding="utf-8") as of:
        print(open_file_name) 
        lines = of.readlines()
    with open(save_file_name, "w", encoding="utf-8") as sf:
        for line in lines:
            word_list = line.split()
            if re.match("EOL", line):
                sf.write("\n")
            if len(word_list)==2:
                if word_list[1] == hold_words_class:
                    sf.write(word_list[0]+" ")
                else:
                    word_list[1] = re.sub("詞","",word_list[1])
                    sf.write(word_list[1]+" ")
