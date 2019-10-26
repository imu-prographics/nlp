cd C:\develop\seq2seq\data
call C:\ProgramData\Miniconda3\Scripts\activate.bat nlp
:: python select_main_contents.py
:: python delete_noise.py
:: python new_line.py
:: python  mecab_part.py
::python .\delete_needless_lines.py
python .\create_masked_corpus.py -m 助詞 -s particle -r 0.3
python .\merge_data.py .\text\masked\ _masked_particle
python .\cut_seq_to_avg_len.py .\text\masked\merged_masked_particle.txt .\text\cut_masked_particle.txt
python replace_to_unk.py .\text\cut_masked_particle.txt .\text\unk_masked_particle.txt
pause