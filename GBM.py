import os

import lightgbm as lgb
import numpy as np
import json

__random_sate__ = 41

with open('GBM_params.json', 'r') as f:
    gbm_para = json.load(f)

gbm = lgb.LGBMClassifier(max_bin=31, learning_rate=0.01, n_estimators=10, random_state=__random_sate__)


def train(train_set, training_sample_weight=None):
    lgb_train = lgb.Dataset(train_set[0], label=train_set[1])
    gbm.fit(train_set[0], train_set[1], eval_metric='auc', sample_weight=training_sample_weight)
    return gbm


def test(test_set):
    lgb_test = lgb.Dataset(test_set[0], label=test_set[1])
    y_pred = gbm.predict(lgb_test, num_iteration=gbm.best_iteration)
    return y_pred


"""
prefix: (str) the string of the prefix of the file of the array in cache 
"""


def ndarray_to_cache(path, prefix, year, array):
    with open(path + prefix + str(year) + '-data.txt', 'wb') as file:
        np.savetxt(file, array)


def cache_to_ndarray(path, prefix, year):
    with open(path + prefix + str(year) + '-data.txt', 'rb') as file:
        return np.load(file)


def set_weight(path, prefix, year, weight_array):
    with open(path + prefix + str(year) + '-data.txt.weight', 'wb') as file:
        np.save(file, weight_array)
