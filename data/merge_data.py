# coding: utf-8
import glob

files = glob.glob("./nucc/*.txt")
with open("./corpus.txt","w",encoding="utf-8") as merged_file:
    for filename in files:
        with open(filename,"r",encoding="utf-8") as f:
            data = f.read()
            merged_file.write(data)