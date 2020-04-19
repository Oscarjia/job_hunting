# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import os
import numpy as np
import pandas as pd
from utils import get_stopwords, segmentData, create_vocab, read_vocab, tokenizer, onehot
import json
import tensorflow as tf


# 数据预处理
def processing_data(infile, labelfile, outfile, vocab_file, stopwords_file):
    print('Loading stopwords...')
    stopwords = get_stopwords(stopwords_file)

    print('Loading data...')
    data = pd.read_csv(infile)

    print('Saving labels')
    with open(labelfile, 'w') as f:
        for label in data.columns[2:]:
            f.write(label + '\n')

    # 把句子分割成词
    print('Splitting content')
    contents = data['content'].tolist()
    seg_contents = segmentData(contents, stopwords)

    if not os.path.exists(vocab_file):
        print('Creating vocabulary...')
        create_vocab(seg_contents, vocab_file, 50000)

    print('Loading vocabulary...')
    w2i, _ = read_vocab(vocab_file)

    # word2id
    print('Tokenize...')
    token_contents = [tokenizer(c, w2i) for c in seg_contents]
    data['content'] = token_contents

    # 把标签转换成one hot形式
    print('One-hot label')
    for col in data.columns[2:]:
        label = data[col].tolist()
        onehot_label = [onehot(l) for l in label]
        data[col] = onehot_label

    print('Saving...')
    data[data.columns[1:]].to_csv(outfile, index=False)


# 将数据集分割成训练集和验证集
def split_train_valid(infile, trainfile, validfile):
    unsplit = pd.read_csv(infile)
    # shuffle
    unsplit = unsplit.sample(frac=1.0)

    valid = unsplit.iloc[0:5000]
    train = unsplit.iloc[5000:]
    valid.to_csv(validfile, index=False)
    train.to_csv(trainfile, index=False)


# 将数据用dataset格式进行包装
def prepare_data(file_path):
    csv_data = pd.read_csv(file_path)
    size = csv_data.shape[0]
    x = np.zeros((size, 1000))
    for i, c in enumerate(csv_data[csv_data.columns[0]].tolist()):
        x[i,:] = np.array(json.loads(c))
    y = np.zeros((size, 20, 4))
    for i, col in enumerate(csv_data.columns[1:]):
        y[:, i, :] = np.array([json.loads(l) for l in csv_data[col].tolist()])
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset