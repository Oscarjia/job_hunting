# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

from flask import Flask, request, render_template
import argparse
import pandas as pd
import random
from extract_opinion import extractOpinion


def add_arguments(parser):
    """
    building ArgumentPaster
    """
    parser.add_argument("--news_file", type=str, default='data/testing_news.csv', help="the file of news corpus")
    parser.add_argument("--word_list", type=str, default='data/word_speak.txt', help='the target verb list')
    parser.add_argument("--cws_model", type=str, default='ltp_data_v3.4.0/cws.model', help="cws model")
    parser.add_argument("--pos_model", type=str, default='ltp_data_v3.4.0/pos.model', help="pos model")
    parser.add_argument("--par_model", type=str, default='ltp_data_v3.4.0/parser.model', help="parser model")
    parser.add_argument("--ner_model", type=str, default='ltp_data_v3.4.0/ner.model', help="ner model")


# 加载测试用语料
def get_news(path):
    news = pd.read_csv(path, encoding='gb18030')
    return news['content'].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()

    news = get_news(flags.news_file)
    eo = extractOpinion(flags.cws_model, flags.pos_model, flags.par_model, flags.ner_model, flags.word_list)

    app = Flask(__name__)


    @app.route('/alex/project4/', methods=['POST', 'GET'])
    def show():
        if request.method == 'POST':
            corpus = request.form.get('corpus')
        else:
            corpus = random.choice(news)
        result = eo.extract(corpus)
        return render_template('double.html', corpus=corpus, result=result)


    app.run(host='0.0.0.0', port=8888)