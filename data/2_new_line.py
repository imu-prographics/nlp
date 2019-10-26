import glob
import re
import os

read_dir = "./text/noise_dlt/"
save_dir = "./text/indention/"
flag_maru = False
flag_kagi = False
read_paths = glob.glob(read_dir + "*.txt")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    print(save_path)
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    with open(save_path,"w",encoding="utf-8") as sf:
        for line in lines:
            tokens = list(line)
            for i, token in enumerate(tokens):
                sf.write(token)
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
                        sf.write(' \n')
                    if token == '？':
                        sf.write('\n')