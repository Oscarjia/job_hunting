python3 ../web_main.py \
--label_file=../scripts/data/labels.txt \
--vocab_file=../scripts/data/vocab.txt \
--data_file=../scripts/data/sentence.txt \
--test_sentence_file=../scripts/data/test_sentences.csv \
--out_file=../scripts/data/sentence_out.json \
--prob=False \
--batch_size=300 \
--feature_num=20 \
--checkpoint_dir=../scripts/data/elmo_ema_0120