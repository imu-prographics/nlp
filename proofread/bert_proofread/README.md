# MLM only BERT

Masked Language Modelのみで事前学習を行えるBERTの実装。<br>
以下のリポジトリのコードを改変して作成<br>
https://github.com/Kosuke-Szk/ja_text_bert

# 使い方
1. 単一のテキストファイルにまとめたコーパスを準備

2. `jupyter notebook` か `jupyter lab` の環境を実行

3. `text_preprocess.ipynb` を実行して、データをTrain, Valid, Test用に分割

4. `MLM_only_bert.ipynb` を実行
