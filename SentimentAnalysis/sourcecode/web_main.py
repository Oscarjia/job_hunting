import argparse
import json
import time

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, redirect, url_for, render_template
from dataset import DataSet
from model import Model
from utils import *
from data_preprocess_web import process_data

from pyecharts.charts import Radar
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

from datetime import timedelta

from selenium.webdriver.chrome.options import Options


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # mode
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")

    # data
    parser.add_argument("--data_file", type=str, nargs='+', default=None, help="data file for train or inference")
    parser.add_argument("--label_file", type=str, default=None, help="label file")
    parser.add_argument("--vocab_file", type=str, default=None, help="vocab file")
    parser.add_argument("--embed_file", type=str, default=None, help="embedding file to restore")
    parser.add_argument("--out_file", type=str, default=None, help="output file for inference")
    parser.add_argument("--test_sentence_file", type=str, default=None, help="file contains 10 test comments")
    parser.add_argument("--split_word", type='bool', nargs="?", const=True, default=True,
                        help="Whether to split word when oov")
    parser.add_argument("--max_len", type=int, default=1200, help='max length for doc')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--reverse", type='bool', nargs="?", const=True, default=False, help="Whether to reverse data")
    parser.add_argument("--prob", type='bool', nargs="?", const=True, default=False, help="Whether to export prob")

    # model
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--decay_schema", type=str, default='hand', help='learning rate decay: exp | hand')
    parser.add_argument("--encoder", type=str, default='gnmt', help="gnmt | elmo")
    parser.add_argument("--decay_steps", type=int, default=10000, help="decay steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. RMS: 0.001 | 0.0001")
    parser.add_argument("--focal_loss", type=float, default=2., help="gamma of focal loss")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="embedding_dropout")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="drop out keep ratio for training")
    parser.add_argument("--weight_keep_drop", type=float, default=0.8, help="weight keep drop")
    parser.add_argument("--l2_loss_ratio", type=float, default=0.0, help="l2 loss ratio")
    parser.add_argument("--rnn_cell_name", type=str, default='lstm', help='rnn cell name')
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding_size")
    parser.add_argument("--num_units", type=int, default=300, help="num_units")
    parser.add_argument("--double_decoder", type='bool', nargs="?", const=True, default=False,
                        help="Whether to double decoder size")
    parser.add_argument("--variational_dropout", type='bool', nargs="?", const=True, default=True,
                        help="Whether to use variational_dropout")

    # clf
    parser.add_argument("--target_label_num", type=int, default=4, help="target_label_num")
    parser.add_argument("--feature_num", type=int, default=20, help="feature_num")

    # predict
    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/visual-semantic',
                        help="checkpoint dir to save model")


def convert_to_hparams(params):
    hparams = tf.contrib.training.HParams()
    for k, v in params.items():
        hparams.add_hparam(k, v)
    return hparams


def predict(flags, model, sentence):
    cnt = 0
    # process sentence
    process_data(flags.data_file, labels, sentence)
    # transform to dataset
    dataset = DataSet(flags.data_file, flags.vocab_file, flags.label_file, flags.batch_size, reverse=flags.reverse,
                      split_word=flags.split_word, max_len=flags.max_len)

    for (source, lengths, _, ids) in dataset.get_next(shuffle=False):
        predict, logits = model.inference_clf_one_batch(sess, source, lengths)
        for i, (p, l) in enumerate(zip(predict, logits)):
            for j in range(flags.feature_num):
                label_name = dataset.i2l[j]
                if flags.prob:
                    tag = [float(v) for v in l[j]]
                else:
                    tag = dataset.tag_i2l[np.argmax(p[j])]
                dataset.items[cnt + i][label_name] = tag
        cnt += len(lengths)

    return dataset.items[0]


def tag2label(num):
    num = int(num)
    if num == 1:
        return '正面情感'
    elif num == 0:
        return '中性情感'
    elif num == -1:
        return '负面情感'
    elif num == -2:
        return '未提及'
    else:
        return '出错了'

def render(html, item):
    comment = item['sentence']
    ltc = tag2label(item['location_traffic_convenience'])
    ldfbd = tag2label(item['location_distance_from_business_district'])
    letf = tag2label(item['location_easy_to_find'])
    swt = tag2label(item['service_wait_time'])
    swa = tag2label(item['service_waiters_attitude'])
    spc = tag2label(item['service_parking_convenience'])
    sss = tag2label(item['service_serving_speed'])
    pl = tag2label(item['price_level'])
    pce = tag2label(item['price_cost_effective'])
    pd = tag2label(item['price_discount'])
    ed = tag2label(item['environment_decoration'])
    en = tag2label(item['environment_noise'])
    es = tag2label(item['environment_space'])
    ec = tag2label(item['environment_cleaness'])
    dp = tag2label(item['dish_portion'])
    dt = tag2label(item['dish_taste'])
    dl = tag2label(item['dish_look'])
    dr = tag2label(item['dish_recommendation'])
    ooe = tag2label(item['others_overall_experience'])
    owtca = tag2label(item['others_willing_to_consume_again'])
    return render_template(html, comment=comment, ltc=ltc, ldfbd=ldfbd, letf=letf, swt=swt, swa=swa, spc=spc, sss=sss,
                           pl=pl, pce=pce, pd=pd, ed=ed, en=en, es=es, ec=ec, dp=dp, dt=dt, dl=dl, dr=dr, ooe=ooe,
                           owtca=owtca)


def get_test_sentence(file):
    s_csv = pd.read_csv(file)
    return s_csv['content'].tolist()


def create_radarmap(item):
    values = [[]]
    inds = []
    d = {
        'location_traffic_convenience': '交通是否便利',
        'location_distance_from_business_district': '距离商圈远近',
        'location_easy_to_find': '是否容易寻找',
        'service_wait_time': '排队等候时间',
        'service_waiters_attitude': '服务人员态度',
        'service_parking_convenience': '是否容易停车',
        'service_serving_speed': '点菜/上菜速度',
        'price_level': '价格水平',
        'price_cost_effective': '性价比',
        'price_discount': '折扣力度',
        'environment_decoration': '装修情况',
        'environment_noise': '嘈杂情况',
        'environment_space': '就餐空间',
        'environment_cleaness': '卫生情况',
        'dish_portion': '分量',
        'dish_taste': '口感',
        'dish_look': '外观',
        'dish_recommendation': '推荐程度',
        'others_overall_experience': '本次消费感受',
        'others_willing_to_consume_again': '再次消费的意愿'
    }
    labels = [label for label in item.keys() if label not in ['id', 'content', 'sentence']]
    for label in labels:
        if int(item[label]) != -2:
            values[0].append(int(item[label]) + 1)
            inds.append(d[label])

    radar = (
        Radar()
            .add_schema(
            schema=[
                opts.RadarIndicatorItem(name=ind, max_=2) for ind in inds
            ]
        )
            .add('分类指标', values)
    )
    make_snapshot(snapshot, radar.render(), "../static/images/radar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()

    options = Options()
    options.add_argument('-headless')
    options.add_argument('-no-sandbox')
    options.add_argument('-disable-dev-shm-usage')


    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        hparams = load_hparams(flags.checkpoint_dir,
                               {"mode": 'inference', 'checkpoint_dir': flags.checkpoint_dir + "/best_eval",
                                'embed_file': None})
        model = Model(hparams)
        model.build()

        try:
            model.restore_model(sess)  # restore best solution
        except Exception as e:
            print("unable to restore model with exception", e)
            exit(1)

        scalars = model.scalars.eval(session=sess)
        print("Scalars:", scalars)
        weight = model.weight.eval(session=sess)
        print("Weight:", weight)
        cnt = 0

        with open(flags.label_file, 'r') as f:
            labels = [label.strip() for label in f.readlines()]

        test_sentence = get_test_sentence(flags.test_sentence_file)

        app = Flask(__name__)


        @app.route('/alex/project2/', methods=['POST', 'GET'])
        def show():
            if request.method == 'POST':
                sentence = request.form.get('comment')
                item = predict(flags, model, sentence)
                item['sentence'] = sentence
                create_radarmap(item)
                return render('single.html', item)
            else:
                sentence = random.choice(test_sentence)
                item = predict(flags, model, sentence)
                item['sentence'] = sentence
                create_radarmap(item)
                return render('single.html', item)

        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

        app.run(host='0.0.0.0', port=8888)
        # app.run(debug=True)