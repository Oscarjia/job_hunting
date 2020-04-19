# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import os
import json
from model import get_model
import tensorflow as tf
import math
import numpy as np
from utils import macro_f1
from tqdm import tqdm


def train(flags, logger, train_dataset, valid_dataset, root_path=''):
    # 读取/初始化检查点参数
    logger.info('Loading checkpoint params...')
    if os.path.exists(root_path + flags.ckpt_params_path):
        with open(root_path + flags.ckpt_params_path, 'r') as f:
            params = json.loads(f.readline())
    else:
        params = {'epoch': 0, 'patience': 1, 'final_learn': 1, 'lr': 1e-3,
                  'pre_best_loss': 10000000, 'pre_best_metrics': (0.0, 0.0, 0.0),
                  'pre_best_ckpt_path': ''}

    # 加载模型
    logger.info('Initialize model...')
    model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                      flags.dropout_loss_rate, flags.label_num)
    if params['pre_best_ckpt_path']:
        model.load_weights(root_path + params['pre_best_ckpt_path'])
    # 选择优化器
    logger.info(f'Setting learning rate as {params["lr"]}')
    optimizer = tf.keras.optimizers.Adam(params['lr'])

    # 设置其他参数
    train_batch_nums = math.ceil(flags.num_train_sample / flags.batch_size)
    while True:
        params['epoch'] += 1

        # 初始化训练参数
        train_losses = 0
        valid_losses = 0
        avg_prec, avg_recall, avg_f1 = 0, 0, 0

        # 训练(train)
        with tqdm(enumerate(train_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)),
                  total=train_batch_nums) as pbar:
            for train_step, batch in pbar:
                x, y = batch
                y_true = [y[:, i, :] for i in range(y.shape[1])]
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss = [tf.keras.losses.categorical_crossentropy(y_i, l_i) for y_i, l_i
                            in zip(y_true, logits)]
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_losses += sum(sum(loss) / x.shape[0])  # 如果num_sample/batch_size不为整数，那么最后一个batch的size不等于batch_size

                if train_step == train_batch_nums - 1:
                    # 验证(valid)
                    logger.info(f'Validating at epoch{params["epoch"]}')
                    for _, batch in enumerate(valid_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)):
                        x, y = batch
                        y_true = [y[:, i, :] for i in range(y.shape[1])]
                        pred = model.predict(x)
                        loss = [tf.keras.losses.categorical_crossentropy(y_i, p_i) for y_i, p_i
                                in zip(y_true, pred)]
                        valid_losses += sum(sum(loss) / x.shape[0])
                        for i in range(x.shape[0]):
                            prec, recall, f1 = macro_f1(4, list(map(np.argmax, np.array(pred)[:, i, :])),
                                                        list(map(np.argmax, y[i])))
                            avg_prec += prec
                            avg_recall += recall
                            avg_f1 += f1

                    valid_losses = valid_losses / (flags.num_valid_sample / flags.batch_size)
                    avg_prec /= flags.num_valid_sample
                    avg_recall /= flags.num_valid_sample
                    avg_f1 /= flags.num_valid_sample

                pbar.set_description(f'Epoch{params["epoch"]}: train loss={train_losses / train_step + 1:.4f}, ' +
                                     f'valid loss={valid_losses:.4f}, ' +
                                     f'prec={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}')
        logger.info(f'At epoch{params["epoch"]}, training loss={train_losses:.4f}')

        # 检查点
        if valid_losses < params['pre_best_loss']:
            logger.info(f'Saving best checkpoint...')
            params['pre_best_loss'] = float(valid_losses)
            params['pre_best_metrics'] = (float(avg_prec), float(avg_recall), float(avg_f1))
            params['pre_best_ckpt_path'] = 'model/ckpt/best_ckpt'
            model.save_weights(root_path + params['pre_best_ckpt_path'])
            # 覆盖之前的最佳检查点参数
            with open(root_path + flags.ckpt_params_path, 'w') as f:
                json.dump(params, f)
            # 记录每次loss降低
            with open(root_path + flags.train_log, 'a') as f:
                f.write(
                    f'At epoch{params["epoch"]}, lr={params["lr"]}, train loss={train_losses / train_batch_nums:.4f}, valid loss{valid_losses:.4f}, precison={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}\n')

            params['patience'] = 1
        else:
            logger.info(f'Loss increased at epoch{params["epoch"]}!')
            if params['patience'] > 0:
                params['patience'] -= 1
            else:
                if params['final_learn'] > 0:
                    logger.info(f'Restore previous best checkpoint...')
                    model.load_weights(root_path + params['pre_best_ckpt_path'])
                    params['final_learn'] -= 1
                    params['lr'] /= 10
                    logger.info(f'Decrease learning rate to {params["lr"]}')
                    optimizer = tf.keras.optimizers.Adam(params['lr'])
                    params['patience'] = 1
                else:
                    model.save_weights(root_path + flags.weight_save_path)
                    logger.info('End of Train.')
                    logger.info(f'Best valid loss: {params["pre_best_loss"]:.4f}, precsion: {params["pre_best_metrics"][0]:.4f}, recall: {params["pre_best_metrics"][1]:.4f}, f1: {params["pre_best_metrics"][2]:.4f}')
                    break


def valid(flags, valid_dataset, root_path=''):
    model = get_model(flags.max_len, flags.vocab_size, flags.embedding_dim, flags.lstm_unit,
                      flags.dropout_loss_rate, flags.label_num)

    model.load_weights(root_path+flags.weight_save_path)

    valid_losses = 0
    avg_prec, avg_recall, avg_f1 = 0, 0, 0

    for _, batch in enumerate(valid_dataset.shuffle(flags.shuffle_size).batch(flags.batch_size)):
        x, y = batch
        y_true = [y[:, i, :] for i in range(y.shape[1])]
        pred = model.predict(x)
        loss = [tf.keras.losses.categorical_crossentropy(y_i, p_i) for y_i, p_i
                in zip(y_true, pred)]
        valid_losses += sum(sum(loss) / x.shape[0])
        for i in range(x.shape[0]):
            prec, recall, f1 = macro_f1(4, list(map(np.argmax, np.array(pred)[:, i, :])),
                                        list(map(np.argmax, y[i])))
            avg_prec += prec
            avg_recall += recall
            avg_f1 += f1

    valid_losses = valid_losses / (flags.num_valid_sample / flags.batch_size)
    avg_prec /= flags.num_valid_sample
    avg_recall /= flags.num_valid_sample
    avg_f1 /= flags.num_valid_sample

    print(f'Valid loss={valid_losses:.4f}, precision={avg_prec:.4f}, recall={avg_recall:.4f}, f1={avg_f1:.4f}')


