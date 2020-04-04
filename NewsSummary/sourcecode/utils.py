# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2019
# ========================================

from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
import re
import jieba


class SIF:
    """SIF(smooth inverse frequency),
    method from A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
    https://openreview.net/pdf?id=SyK00v5xx

    Using word2vec as word embedding.
    """

    def __init__(self, model_path, size, a=1e-5):
        self._model = word2vec.Word2Vec.load(model_path)
        self._size = size  # size of word embedding in word2vec model
        self._a = a  # param a usually take 1e-3 or 1e-5

        # count the total number of words, used for calculate frequency
        self._total = 0
        for k in self._model.wv.vocab.keys():
            self._total += self._model.wv.vocab[k].count

    def change_a(self, a):
        """change the param a"""
        self._a = a

    def get_fre(self, word):
        """get the frequency of given word"""
        return self._model.wv.vocab[word].count / self._total

    def embedding_step1(self, word_list):
        """get the weighted sentence embedding"""
        v = np.zeros(self._size, )
        count = 0
        for word in word_list:
            if word not in self._model.wv: continue
            weighted_wordvec = self._a / (self._a + self.get_fre(word)) * self._model.wv[word]
            v += weighted_wordvec
            count += 1
        return v / count if count else v

    @staticmethod
    def embedding_step2(matrix):
        """subtract the first principle component"""
        pca = PCA(n_components=1)
        pca.fit(matrix)
        pc = pca.components_
        return matrix - matrix.dot(pc.T) * pc

    def embedding(self, sentences):
        """sentence embedding"""
        none_sentence = []
        vectors = []

        # the sentence that doesn't get embedding in step1 shouldn't be caculated in step2
        for i, s in enumerate(sentences):
            # step1
            v = self.embedding_step1(s)
            if v.any():
                vectors.append(v)
            else:
                none_sentence.append(i)
        # step2
        matrix = np.zeros([len(vectors), self._size])
        for i, v in enumerate(vectors):
            matrix[i, :] = v
        matrix = self.embedding_step2(matrix)
        new_vectors = [matrix[i, :] for i in range(matrix.shape[0])]

        # fill the none_sentence
        for index in none_sentence:
            new_vectors.insert(index, np.zeros([self._size, ]))
        return new_vectors


def cut_content(content):
    """将新闻正文切分成单个句子，并筛选掉一些不是正文的句子（比如说图片的注释等），需要用re库

    Args:
        content: 正文内容
    Returns:
        sentences: 列表形式的切分完毕的句子
    """
    # 分段
    paragraphs = [p.strip() for p in re.split(r'[\n\u3000\r]', content) if p.strip()]

    # 分句
    sentences = []
    for p in paragraphs:
        for i in range(len(p)):
            s = ''
            while p:
                if p[0] == '”' and len(s) > 0 and s[-1] in '。？！':
                    s += p[0]
                    p = p[1:]
                    sentences.append(s)
                    s = ''
                else:
                    s += p[0]
                    p = p[1:]
                    if s[-1] in '。？！':
                        sentences.append(s)
                        s = ''
    return sentences


def cut_sentence(sentence, stopwords=[]):
    """用jieba对句子进行分词处理（需要用jieba库）
    """
    return [word for word in jieba.cut(sentence.strip()) if word.strip() if word not in stopwords]


def cosine(v1, v2):
    """计算两个向量的余弦相似度（需要用numpy库）"""
    assert v1.shape == v2.shape
    num = np.dot(v1, v2.T)
    demon = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / demon if demon != 0 else 0