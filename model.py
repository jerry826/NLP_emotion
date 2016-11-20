# -*- encoding=utf8 -*-
"""
@author: Jerry
@contact: lvjy3.15@sem.tsinghua.edu.com
@file: model.py
@time: 2016/11/20 19:13
"""

from utils import *


def data_scale():
    pass


def feature_selection():
    pass


def predict(train, test, type=0):
    pass


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    train_single_set, train_multi_set, test_single_set, test_multi_set = prepare_data()
    test_single_pred = predict(train_single_set, test_single_set)
    test_multi_pred = predict(train_multi_set, test_multi_set)
