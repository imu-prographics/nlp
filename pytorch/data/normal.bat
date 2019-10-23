cd C:\develop\seq2seq\data
call C:\ProgramData\Miniconda3\Scripts\activate.bat nlp
:: python select_main_contents.py
:: python delete_noise.py
:: python new_line.py
:: python  mecab_part.py
::python .\delete_needless_lines.py
::python .\create_normal_corpus.py
::python .\merge_data.py .\text\normal\ _normal
::python .\cut_seq_to_avg_len.py .\text\normal\merged_normal.txt .\text\cut_normal.txt
python replace_to_unk.py .\text\cut_normal.txt .\text\unk_normal.txt
pause