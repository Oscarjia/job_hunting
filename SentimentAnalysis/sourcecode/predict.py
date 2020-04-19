# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

from model import get_model
from utils import get_stopwords, read_vocab, segmentData, tokenizer
import numpy as np


class SentimentAnalysis:
    def __init__(self, flags):
        # 加载模型
        self.model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                               flags.dropout_loss_rate, flags.label_num)
        self.model.load_weights(flags.weight_save_path)
        # 预加载处理评价数据库
        self.stopwords = get_stopwords(flags.stopwords_file)
        self.w2i, _ = read_vocab(flags.vocab_file)
        with open(flags.label_file, 'r') as f:
            self.labels = [l.strip() for l in f.readlines()]
        self.classify = ['Not mention', 'Bad', 'Normal', 'Good']

    # string to tokens
    def process_data(self, comment):
        seg_comment = segmentData([comment], self.stopwords)[0]
        tokens = tokenizer(seg_comment, self.w2i)
        return tokens

    # string to labels
    def predict(self, comment):
        tokens = self.process_data(comment)
        pred = self.model.predict(np.array(tokens).reshape((1, len(tokens))))
        categorys = [np.argmax(p) for p in pred]
        return categorys

    # 打印结果
    def print_result(self, comment):
        categorys = self.predict(comment)
        for c, l in zip(categorys, self.labels):
            print(f'{l:-<44} {self.classify[c]}')