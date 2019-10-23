cd C:\develop\seq2seq\data
call C:\ProgramData\Miniconda3\Scripts\activate.bat nlp
python .\create_min_corpus.py
python .\cut_seq_to_avg_len.py .\text\mai2000a_normal.txt .\text\mai2000a_normal_cut.txt
python .\cut_seq_to_avg_len.py .\text\mai2000a_masked_particle.txt .\text\mai2000a_masked_particle_cut.txt
python .\replace_to_unk.py .\text\mai2000a_normal_cut.txt .\text\mai2000a_normal_unk.txt
python .\replace_to_unk.py .\text\mai2000a_masked_particle_cut.txt .\text\mai2000a_masked_particle_unk.txt
pause