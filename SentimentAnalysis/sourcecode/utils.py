# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import re
import jieba
from collections import Counter


# 读取stopwords
def get_stopwords(file):
    with open(file, 'r') as f:
        stopwords = [s.strip() for s in f.readlines()]
    return stopwords


# 用jieba分词对内容进行切割，并用空格连接分割后的词
def segmentData(contents, stopwords):
    def content2words(content, stopwords):
        content = re.sub('~+', '~', content)
        content = re.sub('\.+', '~', content)
        content = re.sub('～+', '～', content)
        content = re.sub('(\n)+', '\n', content)
        return ' '.join([word for word in jieba.cut(content) if word.strip() if word not in stopwords])

    seg_contents = [content2words(c, stopwords) for c in contents]
    return seg_contents


# 创建字典
def create_vocab(data, vocab_file, vocab_size):
    words = Counter()

    for content in data:
        words.update(content.split())

    special_tokens = ['<UNK>', '<SOS>', '<EOS>']

    with open(vocab_file, 'w') as f:
        for token in special_tokens:
            f.write(token + '\n')
        for token, _ in words.most_common(vocab_size - len(special_tokens)):
            f.write(token + '\n')


# 建立word2id和id2word映射
def read_vocab(vocab_file):
    word2id = {}
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f):
            word = line.strip()
            word2id[word] = i
    id2word = {v:k for k, v in word2id.items()}
    return word2id, id2word


# 将分割后的句子转化为id
def tokenizer(content, w2i, max_token=1000):
    tokens = content.split()
    ids = []
    for t in tokens:
        if t in w2i:
            ids.append(w2i[t])
        else:
            ids.append(w2i['<UNK>'])
    ids = [w2i['<SOS>']] + ids[:max_token-2] + [w2i['<EOS>']]
    ids += (max_token - len(ids)) * [w2i['<EOS>']]
    assert len(ids) == max_token
    return ids


# 将评级转化为onehot形式
def onehot(label):
    onehot_label = [0, 0, 0, 0]
    onehot_label[label+2] = 1
    return onehot_label


# f1
def macro_f1(label_num, predicted, label):
    results = [{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for _ in range(label_num)]
    # 统计true positive, false positive, false negative, true negative
    for i, p in enumerate(predicted):
        l = label[i]
        for j in range(label_num):
            if p == j:
                if l == j:
                    results[j]['TP'] += 1
                else:
                    results[j]['FP'] += 1
            else:
                if l == j:
                    results[j]['FN'] += 1
                else:
                    results[j]['TN'] += 1

    precision = [0.0] * label_num
    recall = [0.0] * label_num
    f1 = [0.0] * label_num
    # 对每一类标签都计算precision, recall和f1, 并求平均
    for i in range(label_num):
        if results[i]['TP'] == 0:
            if results[i]['FP'] == 0 and results[i]['FN'] == 0:
                precision[i] = 1.0
                recall[i] = 1.0
                f1[i] = 1.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0
                f1[i] = 0.0
        else:
            precision[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FP'])
            recall[i] = results[i]['TP'] / (results[i]['TP'] + results[i]['FN'])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return sum(precision) / label_num, sum(recall) / label_num, sum(f1) / label_num







