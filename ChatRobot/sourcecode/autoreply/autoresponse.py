from .searchInDatabase import DatabaseAnswer
from .searchBySpider import WebSpider
from .GenerateByGPT2 import GPT2Generate


class responseGenerate:
    def __init__(self, database_path, w2v_path, stopwords_path, model_path, vocab_path):
        self.DA = DatabaseAnswer(database_path, w2v_path, stopwords_path)
        self.WS = WebSpider()
        self.GPT2G = GPT2Generate(model_path, vocab_path)


    def generate(self, inputQ):
        db_answer = self.DA.getAnswer(inputQ)
        if db_answer: return (db_answer, 'From Database')

        web_answer = self.WS.searchAnswer(inputQ)
        if web_answer[0]: return web_answer

        GPT2_answer = self.GPT2G.generate(inputQ)
        return (GPT2_answer, 'From GPT2')