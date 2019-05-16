#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:16:39 2019

@author: chenjieyang
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_min_data(train_end='2018-05-13 00:00'):
    df = pd.read_csv('data/rebar_mins.csv', header=2)

    def time_parser(t): return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M")
    time = list(map(time_parser, df.time))
    df['time'] = time
    df.set_index('time', inplace=True)
    train_end = datetime.datetime.strptime(train_end, "%Y-%m-%d %H:%M")
    df_train = df.loc[df.index < train_end]
    df_test = df.loc[df.index >= train_end]
    return df_train, df_test


def get_idx(i, n, step):
    x_start = i * step
    x_end = x_start + n
    y_start = x_end
    y_end = y_start + step
    return x_start, x_end, y_start, y_end


def prepare_data(df, n, step, test_size=0.3):
    delta = df.drop('volume', axis=1).pct_change()
    log_volume_delta = np.log(df.volume) - np.log(df.volume.shift(1))
    delta['volume'] = log_volume_delta
    delta = delta.dropna(how='all')
    df = df.iloc[1:, :]
    nrows = delta.shape[0]
    i = 0
    X = []
    y = []
    while True:
        x_start, x_end, y_start, y_end = get_idx(i, n, step)
        if y_end > nrows - 1:
            break
        x = delta.iloc[x_start:x_end, :].values
        x = MinMaxScaler().fit_transform(x) * 255
        X.append(x.astype('int'))
        y.append((df.iloc[y_end, :].close - df.iloc[y_start, :].close) / df.iloc[y_start, :].close)
        i += 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_test = np.expand_dims(X_train, -1), np.expand_dims(X_test, -1)
    y_train, y_test = np.array(y_train) >= 0, np.array(y_test) >= 0
    return X_train, X_test, y_train * 1.0, y_test * 1.0
