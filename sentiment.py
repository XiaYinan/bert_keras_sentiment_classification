#! -*- coding:utf-8 -*-
import os
os.environ["TF_KERAS"]="1" ## to enable tf.keras in keras_bert module

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

maxlen = 100

## bert download path: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'

## get the tokenizer from vocab.txt
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
tokenizer = OurTokenizer(token_dict)

## data download path: https://pan.baidu.com/s/1DoQbki3YwqkuwQUOj64R_g#list/path=%2F
df_data = pd.read_csv(r'G:/BaiduNetdiskDownload/weibo_senti_100k/weibo_senti_100k/weibo_senti_100k.csv')
data = df_data[['review','label']].values


# train test split
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 100 == 1]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 100 == 0]

# data preprocessing, including tokenizing, padding, truncating and batch
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen] # truncating
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []
train_D = data_generator(train_data)
valid_D = data_generator(valid_data)


## define model structure
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)
model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

## model training
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=1,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)

## model saving
model.save('my_model.h5')

