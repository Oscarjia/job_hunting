# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import re
import jieba
from pyltp import Postagger, Parser, Segmentor
from collections import defaultdict


class extractOpinion:
    def __init__(self, cws_model, pos_model, par_model, ner_model, word_list_path):
        # pyltp模型路径
        self.cws_model = cws_model
        self.pos_model = pos_model
        self.par_model = par_model
        self.ner_model = ner_model
        # 表示观点的动词
        with open(word_list_path, 'r') as f:
            self.word_list = [word.strip() for word in f.readlines()]

    @staticmethod
    def get_word_list(sentence, model):
        # 得到分词
        segmentor = Segmentor()
        segmentor.load(model)
        word_list = list(segmentor.segment(sentence))
        segmentor.release()
        return word_list

    @staticmethod
    def get_postag_list(word_list, model):
        # 得到词性标注
        postag = Postagger()
        postag.load(model)
        postag_list = list(postag.postag(word_list))
        postag.release()
        return postag_list

    @staticmethod
    def get_parser_list(word_list, postag_list, model):
        # 得到依存关系
        parser = Parser()
        parser.load(model)
        arcs = parser.parse(word_list, postag_list)
        arc_list = [(arc.head, arc.relation) for arc in arcs]
        parser.release()
        return arc_list

    # 依存关系分析
    def dependency_parsing(self, content):
        word_list = self.get_word_list(content, self.cws_model)
        postag_list = self.get_postag_list(word_list, self.pos_model)
        parser_list = self.get_parser_list(word_list, postag_list, self.par_model)
        result = []
        for i in range(len(word_list)):
            result.append((word_list[i], parser_list[i]))
        return result

    # 抽取主语和谓语
    def extract_verb(self, dp_result):
        result = defaultdict(list)
        for i, (word, (parsing_index, parsing_tag)) in enumerate(dp_result):
            if parsing_tag == 'SBV':
                if dp_result[parsing_index - 1][0] in self.word_list:
                    result[parsing_index - 1].append(i)
        return result

    # 将语料按照标点符号分成短句
    @staticmethod
    def split_sentence(dp_result):
        indexes = []
        length = len(dp_result)
        start = -1
        end = -1
        for i in range(length):
            if start == -1 and dp_result[i][1][1] != 'WP':
                start = i
            if start != -1 and (dp_result[i][0] in '，。！？：' or i == length - 1):
                end = i
                indexes.append((start, end))
                start = -1
                end = -1
        return indexes

    # 找到指向目标谓语的所有宾语所在短句
    @staticmethod
    def find_object(dp_result, verb_index, sentence_index):
        targets = []
        for vob in sorted(verb_index.keys()):
            object_pool = []
            target_sentences = []

            for index, (start, end) in enumerate(sentence_index):
                if index in target_sentences: continue
                for i in range(start, end):
                    if dp_result[i][1] == (vob + 1, 'VOB'):  # word that refers to vob
                        object_pool.append(i)
                        target_sentences.append(index)

            while object_pool:
                vob = object_pool.pop(0)
                for index, (start, end) in enumerate(sentence_index):
                    if index in target_sentences: continue
                    for i in range(start, end + 1):
                        if dp_result[i][1][0] == vob + 1:
                            object_pool.append(i)
                            target_sentences.append(index)
            targets.append(target_sentences)
        return targets

    # 根据谓语位置和宾语所在短句，输出宾语在语料中的具体位置
    @staticmethod
    def get_object_index(verb, sentence_index, obj):
        sentences = []

        for i, s in enumerate(sorted(obj)):
            if sentence_index[s][0] > verb:
                if i != 0:
                    sentences.append((sentence_index[obj[0]][0], sentence_index[obj[i - 1]][1]))
                sentences.append((sentence_index[s][0], sentence_index[obj[-1]][1]))
                break
            elif sentence_index[s][1] > verb:
                if i != 0:
                    sentences.append((sentence_index[obj[0]][0], sentence_index[obj[i - 1]][1]))
                sentences.append((verb + 1, sentence_index[obj[-1]][1]))
                break

        return sentences if sentences else [(sentence_index[obj[0]][0], sentence_index[obj[-1][1]])]

    # 输出主语，谓语和观点
    def extract(self, content):
        content = re.sub(r'\s', '', content)
        dp_result = self.dependency_parsing(content)
        verb_index = self.extract_verb(dp_result)
        sentence_index = self.split_sentence(dp_result)
        objects = self.find_object(dp_result, verb_index, sentence_index)

        opinions = []
        for verb, obj in zip(sorted(verb_index.keys()), objects):
            if not obj: continue
            opinion = []
            sentences = self.get_object_index(verb, sentence_index, obj)
            opinion.append(' '.join([dp_result[svb][0] for svb in verb_index[verb]]))
            opinion.append(dp_result[verb][0])
            if len(sentences) == 1:
                opinion.append(''.join([dp_result[i][0] for i in range(sentences[0][0], sentences[0][1] + 1)]))
            else:
                opinion.append('｜'.join([''.join([dp_result[i][0] for i in range(start, end + 1)]) for (start, end) in sentences]))
            opinions.append(opinion)
        return opinions