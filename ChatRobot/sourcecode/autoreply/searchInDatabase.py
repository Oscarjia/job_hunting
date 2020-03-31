import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import re
import jieba
from collections import defaultdict
from .editDistance import edit_distance


class DatabaseAnswer:
    def __init__(self, database_path, w2v_path, stopwords_path):
        self.database_path = database_path
        self.w2v_path = w2v_path
        with open(stopwords_path, 'r') as f:
            self.stopwords = [word.strip() for word in f.readlines()]
        self.readData()
        self.word2VecWarmup()
        self.invertedIndexWarmup()

    # Read database
    def readData(self):
        data = pd.read_csv(self.database_path)
        data.dropna(axis=0, how='any', inplace=True)
        self.questions = data['question'].tolist()
        self.answers = data['answer'].tolist()

    def word2VecWarmup(self, n_clusters=4):
        self.model = Word2Vec.load(self.w2v_path)
        self.total = 0
        for k in self.model.wv.vocab.keys():
            self.total += self.model.wv.vocab[k].count
        self.question_vectors = [self.sif_step1(self.sentence2words(q)) for q in self.questions]

        # cluster warm up
        self.clf = KMeans(n_clusters=n_clusters)
        self.clf.fit(self.question_vectors)

    def sentence2words(self, sentence):
        return [word for word in jieba.cut(sentence) if word not in self.stopwords if word.strip()]

    def sif_step1(self, sentence, a=1e-3):
        """
        Sentence embedding.
        Args:
            model: word2vec model
            sentence: splitted sentence
        Returns:
            sentence vector
        """
        v_sentence = np.zeros(self.model.wv['算法'].shape)
        count = 0
        for word in sentence:
            if word not in self.model.wv: continue
            word_fre = self.model.wv.vocab[word].count / self.total
            v_word = a / (a + word_fre) * self.model.wv[word]
            v_sentence += v_word
            count += 1
        return v_sentence / count if count > 0 else v_sentence

    @staticmethod
    def calCosine(vector1, vector2):
        num = np.dot(vector1, vector2.T)
        demon = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return num / demon

    def searchByWord2Vec(self, inputQ, n=20):
        inputQ_vector = self.sif_step1(self.sentence2words(inputQ))
        c = self.clf.predict([inputQ_vector])[0]
        c_questions = []
        for i, boo in enumerate(self.clf.labels_ == c):
            if boo:
                c_questions.append((i, self.questions[i], self.question_vectors[i]))
        relative = [(i, question, self.calCosine(q_vector, inputQ_vector)) for (i, question, q_vector)
                    in c_questions]
        return sorted(relative, key=lambda x: x[-1], reverse=True)[:n]

    def invertedIndexWarmup(self):
        word_list = []
        front_index = {}
        self.inverted_index = defaultdict(list)
        # do front index
        for i, question in enumerate(self.questions):
            front_index[i] = set(self.sentence2words(question))
        # do inverted index
        for k in front_index.keys():
            for word in front_index[k]:
                self.inverted_index[word].append(k)

    @staticmethod
    def search_same(list1, list2):
        result = []
        while list1 and list2:
            if list1[0] < list2[0]:
                list1 = list1[1:]
            elif list1[0] == list2[0]:
                result.append(list1[0])
                list1, list2 = list1[1:], list2[1:]
            else:
                list2 = list2[1:]
        return result

    def searchByInvertedIndex(self, inputQ):
        words = list(set([word for word in self.sentence2words(inputQ) if word in self.inverted_index.keys()]))
        index = []
        while words:
            if not index:
                index = self.inverted_index[words[0]]
                words = words[1:]
            else:
                index = self.search_same(index, self.inverted_index[words[0]])
                words = words[1:]
                if index:
                    continue
                else:
                    break
        return [(i, self.questions[i]) for i in index] if index else []

    @staticmethod
    def sortByEditDistance(inputQ, candidates):
        new_candidates = []
        for i, c in candidates:
            d, _ = edit_distance(inputQ, c)
            new_candidates.append((i, c, d))
        return sorted(new_candidates, key=lambda x: x[2])


    def getAnswer(self, inputQ):
        candidates = []

        w2v_result = [(i, q) for i, q, _ in self.searchByWord2Vec(inputQ)]
        candidates += w2v_result

        invertedIndex_result = self.searchByInvertedIndex(inputQ)
        candidates += invertedIndex_result

        sorted_candidates = self.sortByEditDistance(inputQ, candidates)
        if sorted_candidates:
            best_answer = sorted_candidates[0] + (self.answers[sorted_candidates[0][0]],)
            if best_answer[2] <= 2 and len(inputQ) >= best_answer[2] * 2:
                return best_answer[3]
        return []