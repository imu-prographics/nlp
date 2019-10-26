import sys 
import glob
import re
import os

read_dir = "./text/mecab/"
save_dir = "./text/mecab_particle/"
read_paths = glob.glob(read_dir+"*.txt")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
flag = False
line_stack = []
for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    print(save_path)
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    with open(save_path,"w",encoding="utf-8") as sf:
        for line in lines:
            if re.search("助詞|助動詞|動詞",line):
                flag = True
            line = re.sub("-.*\n","\n",line)
            line_stack.append(line)
            if re.match("EOL",line):
                if flag is True:
                    sf.write("\n".join(line_stack))
                    sf.write('\n')
                    flag = False
                line_stack = []


