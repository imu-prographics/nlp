# coding: utf-8
import glob
import sys
args = sys.argv
dir_name = args[1]
tag_name = args[2]

files = glob.glob(dir_name+"*" + tag_name +".txt")
with open(dir_name+"merged" + tag_name + ".txt","w",encoding="utf-8") as merged_file:
    for filename in files:
        with open(filename,"r",encoding="utf-8") as f:
            data = f.read()
            merged_file.write(data)