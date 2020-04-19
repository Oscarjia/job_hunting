# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import os
import argparse
import logging
import pandas as pd
from predict import SentimentAnalysis
from flask import Flask, request, render_template
from pyecharts.charts import Radar
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
import random

from datetime import timedelta

from selenium.webdriver.chrome.options import Options

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 参数
def initial_arguments():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--root_path', type=str, default='', help='the path of main.py')
    parser.add_argument('--raw_data', type=str, default='data/train.csv', help='unprocessed data')
    parser.add_argument('--processed_data', type=str, default='data/processed.csv',
                        help='data after segment and tokenize')
    parser.add_argument('--train_data', type=str, default='data/train_data.csv', help='path of training data file')
    parser.add_argument('--num_train_sample', type=int, default=100000, help='num of train sample')
    parser.add_argument('--valid_data', type=str, default='data/valid_data.csv', help='path of validating data file')
    parser.add_argument('--num_valid_sample', type=int, default=5000, help='num of valid sample')
    parser.add_argument('--label_file', type=str, default='data/label_names.txt', help='path of label name')
    parser.add_argument('--stopwords_file', type=str, default='data/stopwords.txt', help='path of stopwords file')
    parser.add_argument('--vocab_file', type=str, default='data/vocab.txt', help='path of vocabulary file')
    parser.add_argument('--test_comment_file', type=str, default='data/test_comments.csv', help='comments used for testing')

    # model path
    parser.add_argument('--weight_save_path', type=str, default='model/best_weight', help='path of best weights')

    # model params
    parser.add_argument('--max_len', type=int, default=1000, help='max length of content')
    parser.add_argument('--vocab_size', type=int, default=50000, help='size of vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=256, help='embedding size')
    parser.add_argument('--lstm_unit', type=int, default=128, help='unit num of lstm')
    parser.add_argument('--dropout_loss_rate', type=float, default=0.2, help='dropout loss ratio for training')
    parser.add_argument('--label_num', type=int, default=4, help='num of label')

    # train and valid
    parser.add_argument('--train_log', type=str, default='model/train_log.txt', help='path of train log')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--shuffle_size', type=int, default=128, help='the shuffle size of dataset')
    parser.add_argument('--feature_num', type=int, default=20, help='num of feature')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ckpt_params_path', type=str, default='model/ckpt/ckpt_params.json',
                        help='path of checkpoint params')

    flags, unparsed = parser.parse_known_args()
    return flags


# 日志
def initial_logging(logging_path='info.log'):
    logger = logging.getLogger((__name__))
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=logging_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


# 标签转化成文字
def label2tag(num):
    num = int(num)
    if num == 3:
        return '正面情感'
    elif num == 2:
        return '中性情感'
    elif num == 1:
        return '负面情感'
    elif num == 0:
        return '未提及'
    else:
        return '出错了'


# 输出html
def render(html, comment, labels):
    ltc = label2tag(labels[0])
    ldfbd = label2tag(labels[1])
    letf = label2tag(labels[2])
    swt = label2tag(labels[3])
    swa = label2tag(labels[4])
    spc = label2tag(labels[5])
    sss = label2tag(labels[6])
    pl = label2tag(labels[7])
    pce = label2tag(labels[8])
    pd = label2tag(labels[9])
    ed = label2tag(labels[10])
    en = label2tag(labels[11])
    es = label2tag(labels[12])
    ec = label2tag(labels[13])
    dp = label2tag(labels[14])
    dt = label2tag(labels[15])
    dl = label2tag(labels[16])
    dr = label2tag(labels[17])
    ooe = label2tag(labels[18])
    owtca = label2tag(labels[19])
    return render_template(html, comment=comment, ltc=ltc, ldfbd=ldfbd, letf=letf, swt=swt, swa=swa, spc=spc, sss=sss,
                           pl=pl, pce=pce, pd=pd, ed=ed, en=en, es=es, ec=ec, dp=dp, dt=dt, dl=dl, dr=dr, ooe=ooe,
                           owtca=owtca)


# 获取测试数据
def get_test_comment(file):
    s_csv = pd.read_csv(file)
    return s_csv['content'].tolist()


# 绘制雷达图
def create_radarmap(labels):
    values = [[]]
    inds = []
    d = ['交通是否便利', '距离商圈远近', '是否容易寻找', '排队等候时间', '服务人员态度', '是否容易停车', '点菜/上菜速度', '价格水平',
         '性价比', '折扣力度', '装修情况', '嘈杂情况', '就餐空间', '卫生情况', '分量', '口感', '外观', '推荐程度', '本次消费感受',
         '再次消费的意愿']

    for i in range(len(labels)):
        if labels[i] > 0:
            values[0].append(int(labels[i])-1)
            inds.append(d[i])

    radar = (
        Radar()
            .add_schema(
            schema=[
                opts.RadarIndicatorItem(name=ind, max_=2) for ind in inds
            ]
        )
            .add('分类指标', values)
    )
    make_snapshot(snapshot, radar.render(), "static/images/radar.png")


def main():
    flags = initial_arguments()
    logger = initial_logging()

    options = Options()
    options.add_argument('-headless')
    options.add_argument('-no-sandbox')
    options.add_argument('-disable-dev-shm-usage')

    logger.info('Initialize model')
    sa = SentimentAnalysis(flags)

    test_comment = get_test_comment(flags.test_comment_file)

    app = Flask(__name__)

    @app.route('/alex/project2/', methods=['POST', 'GET'])
    def show():
        if request.method == 'POST':
            # get comment
            comment = request.form.get('comment')
            logger.info(f'Received a comment:\n {comment}')
            labels = sa.predict(comment)
        else:
            # use a random test comment
            comment = random.choice(test_comment)
            labels = sa.predict(comment)

        create_radarmap(labels)
        return render('single.html', comment, labels)

    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

    app.run(host='0.0.0.0', port=8888)


if __name__ == '__main__':
    main()