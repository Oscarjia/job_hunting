# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2019
# ========================================

from summaryModel import ArticleSummary
from flask import Flask, request, render_template
import random
import argparse
import logging


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.add_argument('--model_path', type=str, default='data/model_191115_1', help='word2vec模型路径')
    parser.add_argument('--model_size', type=int, default=250, help='word2vec词向量size')
    parser.add_argument('--a', type=float, default=1e-5, help='SIF模型参数，一般选取1e-5或1e-4')
    parser.add_argument('--stopwords_path', type=str, default='data/哈工大停用标点表.txt', help='停用词文件路径')
    parser.add_argument('--length_lmt', type=int, default=5, help='最短句子长度限制')
    parser.add_argument('--weight', type=float, default=0.4, help='计算相关性时正文和标题的权重')
    parser.add_argument('--times', type=int, default=1, help='平滑的重复次数')
    parser.add_argument('--rg', type=int, default=2, help='滑窗口大小')
    parser.add_argument('--topn', type=int, default=3, help='摘要的句子数量')
    parser.add_argument('--logging_path', type=str, default='data/info.log', help='日志路经')
    parser.add_argument('--example_file', type=str, default='data/example.txt', help='示例路径')


def get_examples(file_path):
    examples = []
    title = ''
    content = ''
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            if line == '\n':
                examples.append((title.strip(), content))
                title = ''
                content = ''
            else:
                if title:
                    content += line
                else:
                    title = line
            line = f.readline()
    return examples



if __name__ == '__main__':
    # 创建解析
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, _ = parser.parse_known_args()

    # 初始化日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=flags.logging_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('日志系统初始化成功！')

    # 初始化模型
    logger.info('正在初始化模型...')
    summary_model = ArticleSummary(flags.model_path, flags.model_size, flags.a, flags.stopwords_path,
                                   flags.length_lmt, flags.weight, flags.times, flags.rg, flags.topn)
    logger.info('模型初始化成功！')

    # 读取示例
    examples = get_examples(flags.example_file)
    logger.info('读取示例成功！')

    app = Flask(__name__)

    @app.route('/alex/newssummary/', methods=['POST', 'GET'])
    def show():
        if request.method == 'POST':
            title = request.form.get('title')
            content = request.form.get('content')
            logger.info('收到用户输入的信息。')
        else:
            title, content = random.choice(examples)
            logger.info('随机展示。')
        summary = summary_model.get_summary(title, content)
        logger.info('成功获取摘要')
        return render_template('summary.html', title=title, content=content, summary=summary)

    app.run(host='0.0.0.0', port=5080)