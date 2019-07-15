import glob
import re

read_dir = "./text/main_extracted/"
save_dir = "./text/noise_dlt/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
read_paths = glob.glob(read_dir + "*_main.txt")
patterns = ["【.*】",
            "◆",
            "★",
            "■",
            "□"
            "◇",
            "◇.*\n",
            "＊",
            "（.*）",
            "＜.*＞",
            "——",
            "▽.*\n",
            "▼",
            "＝.*＝",
            "×××",
            "●.*\n",
            "☆",
            "^[^　].*\n",
            "×",
            "ｈｔｔｐ.*／",
            "^　[Ａ-Ｚ]　",
            "^　一、",
            "^[０-９]　.*\n",
            "^.*[０-９]{5,}.*\n",
            "^.{1,20}\n",
            "^.*[０-９]{1,5}　{1,10}[０-９]{1,5}.*\n",
            "^［.*\n",
            "^.*[０-９]{3,4}・[０-９]{3,4}.*\n",
            "[０-９]{2,4}・[０-９]{2,4}・[０-９]{2,4}",
            "^　.{1,5}　",
            "^　[０-９]月.[０-９]日　",
            "^　◎",
            "^　.*委員長　",
            "Ｑ[０-９]．",
            "》",
            "《"]

for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    save_path = re.sub("_main.txt","",save_path)+"_noise_dlt.txt"
    print(save_path)
    with open(read_path,"r",encoding="utf-8") as rf:
        lines = rf.read().split('\n')
    with open(save_path,"w",encoding="utf-8") as sf:
        lines = of.readlines()
        for line in lines:
            for i in range(len(patterns)):
                line = re.sub(patterns[i],"",line)
                line = re.sub("　{2-30}","　",line)
                line = re.sub("▲","。",line)
            
            m = re.match("^　.*\n",line)
            if m:
                line = re.sub("^△","",line)
                for i in range(10):
                    line = re.sub("^.*・[０-９]{2,}、[０-９]{2,}年.*\n","",line)
                    line = re.sub("^.*　{2,}.*\n","",line)
                    line = re.sub("^　","",line)
                    line = re.sub("^　.*\n","",line)
                    line = re.sub("^\n","",line)
                sf.write(line)