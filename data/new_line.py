import glob
import re
import os

read_dir = "./text/noise_dlt/"
save_dir = "./text/indention/"
read_paths = glob.glob(read_dir + "*_noise_dlt.txt")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    save_path = re.sub("_noise_dlt.txt","",save_path) +"_indention.txt"
    print(save_path)
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    with open(save_path,"w",encoding="utf-8") as sf:
        for line in lines:
            tokens = list(line)
            for i, token in enumerate(tokens):
                sf.write(token)
                if token == '。':
                    sf.write(' \n')