# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import os
import argparse
import logging
from dataprocessing import processing_data, split_train_valid, prepare_data
from train import train

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


def main():
    flags = initial_arguments()
    logger = initial_logging()

    # 处理数据
    if not os.path.exists(flags.train_data):
        logger.info('Processing raw data...')
        processing_data(flags.raw_data, flags.label_file, flags.processed_data, flags.vocab_file, flags.stopwords_file)
        logger.info('Split data to train and valid...')
        split_train_valid(flags.processed_data, flags.train_data, flags.valid_data)

    logger.info('Prepare data...')
    train_dataset = prepare_data(flags.train_data)
    valid_dataset = prepare_data(flags.valid_data)

    # 训练
    logger.info('Start training...')
    train(flags, logger, train_dataset, valid_dataset, flags.root_path)

    logger.info('Finishing training...')


if __name__ == '__main__':
    main()