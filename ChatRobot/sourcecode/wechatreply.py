# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

from flask import Flask, request, make_response
import hashlib
import xml.etree.ElementTree as ET
import time
from autoreply.autoresponse import responseGenerate
import argparse
import logging


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--database_path', type=str, default='data/qa_corpus.csv', help='the path of database file')
parser.add_argument('--w2v_path', type=str, default='model/w2v/model_200219', help='the path of word2vec model')
parser.add_argument('--stopwords_path', type=str, default='data/哈工大停用词表扩展.txt', help='the path of stopwords file')
parser.add_argument('--model_path', type=str, default='model/gpt2/', help='the path of gpt2 model')
parser.add_argument('--vocab_path', type=str, default='data/vocab_small.txt', help='the path of vocabulary file')
parser.add_argument('--logging_path', type=str, default='data/info.log', help='the path of logging file')
parser.add_argument('--record_path', type=str, default='data/dialogue/', help='the path of dialogue file')
flags, unparsed = parser.parse_known_args()

# initial logging
logger = logging.getLogger((__name__))
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=flags.logging_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

logger.info('Initial response generator')
rg = responseGenerate(flags.database_path, flags.w2v_path, flags.stopwords_path, flags.model_path, flags.vocab_path)


def record_dialogue(file, dialogue):
    with open(file, 'a') as f:
        f.write(dialogue)


app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def wechat():
    """
    Auto-reply robot
    """
    # wechat robot
    if request.method == 'GET':
        token = 'Alex'
        signature = request.args.get('signature')
        timestamp = request.args.get('timestamp')
        nonce = request.args.get('nonce')
        echostr = request.args.get('echostr')

        sign = hashlib.sha1(''.join(sorted([token, timestamp, nonce])).encode('utf-8')).hexdigest()
        return echostr if sign == signature else None
    if request.method == 'POST':
        xml_data = ET.fromstring(request.data)
        ToUserName = xml_data.find('ToUserName').text
        FromUserName = xml_data.find('FromUserName').text
        Content = xml_data.find('Content').text
        MsgType = xml_data.find('MsgType').text
        logger.info('Received Data')
        logger.info('User:{}, Content:{}'.format(FromUserName, Content))

        if MsgType != 'text':
            response = '目前只能接受文字哦！'
        else:
            record_dialogue(flags.record_path+FromUserName+'.txt', 'User:'+Content+'\n')
            logger.info('Generating answer...')
            start = time.time()
            response, source = rg.generate(Content)
            end = time.time()
            record_dialogue(flags.record_path + FromUserName + '.txt', 'Reply:' + response + '\n')
            response = 'Reply:{}\nSource:{}\nTime:{:.3f}'.format(response, source, end-start)
        reply = f'<xml><ToUserName><![CDATA[{FromUserName}]]></ToUserName>' \
                f'<FromUserName><![CDATA[{ToUserName}]]></FromUserName>' \
                f'<CreateTime>{str(int(time.time()))}</CreateTime>' \
                f'<MsgType><![CDATA[text]]></MsgType>' \
                f'<Content><![CDATA[{response}]]></Content></xml>'
        logger.info('Sending message...')
        response = make_response(reply)
        response.content_type = 'application/xml'
        return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)

