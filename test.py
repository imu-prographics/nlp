data_path = "./conversation.txt"
num_samples = 1000  # Number of samples to train on.
input_texts = []
target_texts = []
input_words = set()  # 追記
target_words = set()  # 追記

with  open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')  # 行ごとのリストに

for index, line in enumerate(lines[: min(num_samples, len(lines) - 1)]):
    input_text = line
    target_text = lines[index+1]
    # \tが開始記号で\nが終端記号とする
    #target_text = '\t' + target_text + '\n'
    target_text = '\t ' + target_text + ' \n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    words = []
    
    words = input_text.split(' ')
    for word in words:
        if word not in input_words:
            input_words.add(word)
    words = target_text.split(' ')
    for word in words:
        if word not in target_words:
            target_words.add(word)
for txt in input_texts:
    print(txt.replace(' ','') )