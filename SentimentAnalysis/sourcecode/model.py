# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from att import Attention


def get_model(max_len, vocab_size, embedding_dim, lstm_unit, dropout_keep_rate, label_num, show_structure=False):
    inputs = Input((max_len,), name='input')
    embedding = Embedding(vocab_size, embedding_dim, name='embedding')(inputs)
    bilstm1 = Bidirectional(LSTM(lstm_unit, return_sequences=True), name='bi-lstm1')(embedding)
    dropout1 = Dropout(dropout_keep_rate)(bilstm1)
    bilstm2 = Bidirectional(LSTM(lstm_unit, return_sequences=True), name='bi-lstm2')(dropout1)
    dropout2 = Dropout(dropout_keep_rate)(bilstm2)
    att = Attention(max_len, name='attention')(dropout2)
    d_list = [Dense(name=f'dense{i}', units=label_num, activation='softmax')(att) for i in range(20)]

    model = Model(inputs=inputs, outputs=d_list)
    if show_structure:
        model.summary()

    return model
