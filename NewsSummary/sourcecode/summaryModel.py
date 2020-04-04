# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2019
# ========================================

from utils import SIF, cut_content, cut_sentence, cosine


class ArticleSummary:
    """新闻摘要模型，输入标题（非必须，但是可以提高精度）和正文内容，输出由正文若干关键句子组成的摘要"""

    def __init__(self, model_path, model_size, a, stopwords_path, length_lmt,
                 weight, times, rg, topn):
        """初始化模型
        Args:
            model_path: word2vec模型路径
            model_size: word2vec词向量size
            a: SIF模型参数
            stopwords_path: 停用词文件路径
            length_lmt: 最短句子长度限制
            weight: 计算相关性时正文和标题的权重
            times: 平滑的重复次数
            rg: 平滑窗口大小
            topn: 摘要的句子数量
        """
        # 初始化SIF模型
        self.sif = SIF(model_path, model_size, a)
        # 读取停用词
        with open(stopwords_path, 'r') as f:
            self.stopwords = [word.strip() for word in f.readlines() if word.strip()]

        self.length_lmt = length_lmt
        self.weight = weight
        self.times = times
        self.rg = rg
        self.topn = topn

    def relevance(self, title, content, sentences):
        """计算各个句子和标题以及正文的相关性"""
        c = []
        for s in sentences:
            c.append(self.weight * cosine(s, content) + (1 - self.weight) * cosine(s, title))
        return c

    def smooth(self, lst):
        """对相关性进行平滑处理"""
        # 防止index error，对左边界进行限定
        def left_border(i):
            return i - self.rg if i >= self.rg else 0

        for _ in range(self.times):
            smooth_list = []
            for i in range(len(lst)):
                smooth_area = lst[left_border(i):i + self.rg + 1]
                smooth_list.append(sum(smooth_area) / len(smooth_area))
        return smooth_list

    def get_summary(self, title, content):
        """接受输入，并输出摘要"""

        # 将新闻正文切分成一个个单独的句子，并筛选掉长度不高超过5的短句
        sentences = [s for s in cut_content(content) if len(s) > self.length_lmt]

        # 将标题、正文和每个句子进行分词
        cut_list = [cut_sentence(s, self.stopwords) for s in [title, ''.join(sentences)] + sentences]

        # 获得句向量
        vectors = self.sif.embedding(cut_list)

        # 计算相似度
        correlation = self.relevance(vectors[0], vectors[1], vectors[2:])

        # 对相似度进行平滑处理，增加结果摘要的句子连贯性
        new_cor = self.smooth(correlation)

        # 加上序列标记，方便后续按原文顺序输出句子
        new_cor = [(i, c) for i, c in enumerate(new_cor)]

        # 选出最相关的n个句子
        topn_cor = sorted(new_cor, key=lambda x: x[1], reverse=True)[:self.topn]

        # 按原文顺序组成摘要
        summary = ''.join(sentences[i] for i, _ in sorted(topn_cor, key=lambda x: x[0]))

        return summary
