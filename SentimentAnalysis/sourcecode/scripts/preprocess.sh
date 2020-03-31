#!/bin/bash

# Modify the following values depend on your environment
# Path to the csv files
TRAIN_FILE=data/train.csv
VALIDATION_FILE=data/dev.csv
TESTA_FILE=data/test.csv

# Path to pretrained embedding file
EMBEDDING_FILE=data/sgns.sogou.word

VOCAB_SIZE=50000

# Create a folder to save training files

echo 'Process training file ...'
python3 data_preprocess.py \
    --data_file=$TRAIN_FILE \
    --output_file=data/train.json \
    --vocab_file=data/vocab.txt \
    --stopwords_file=data/stopwrods.txt \
    --vocab_size=$VOCAB_SIZE

echo 'Process validation file ...'
python3 data_preprocess.py \
    --data_file=$VALIDATION_FILE \
    --stopwords_file=data/哈工大停用词表扩展.txt \
    --output_file=data/validation.json

echo 'Process testa file ...'
python3 data_preprocess.py \
    --data_file=$TESTA_FILE \
    --stopwords_file=data/哈工大停用词表扩展.txt \
    --output_file=data/testa.json


echo 'Get pretrained embedding ...'
python3 data_preprocess.py \
    --data_file=$EMBEDDING_FILE \
    --output_file=data/embedding.txt \
    --vocab_file=data/vocab.txt \
    --embedding=True

echo "Get label file ..."
cp ../labels.txt data/labels.txt