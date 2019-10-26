import glob
import re

read_dir = "./text/original/"
save_dir = "./text/pick_article/"
read_paths = glob.glob(read_dir + "*.txt")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
patterns = ["＼ＨＯＮ＼","＼Ｔ２＼"]

for read_path in read_paths:
    save_path = save_dir + os.path.basename(read_path)
    print(save_path)
    with open(read_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(save_path, "w", encoding="utf-8") as f:
        for line in lines:
            for i in range(len(patterns)):
                m = re.match(patterns[i],line)
                if m:
                    line = re.sub("＼ＨＯＮ＼","",line)
                    line = re.sub("＼Ｔ２＼","",line)
                    f.write(line)
    