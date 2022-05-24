import lightgbm as lgb
import numpy as np
import json

__random_sate__ = 42

with open('/workspace/FYP/codespace/config/GBM_params.json', 'r') as f:
    gbm_para = json.load(f)


class GBM:

    def __init__(self):
        self.classifier_machine = lgb.LGBMClassifier(learning_rate=0.01, n_estimators=150, random_state=42)

    def train(self, train_set, training_sample_weight=None):
        lgb_train = lgb.Dataset(train_set[0], label=train_set[1])
        self.classifier_machine.fit(train_set[0], train_set[1], eval_metric='auc', sample_weight=training_sample_weight)

    def test(self, test_set):
        lgb_test = lgb.Dataset(test_set[0], label=test_set[1])
        y_pred = self.classifier_machine.predict(lgb_test, num_iteration=self.classifier_machine.best_iteration_)
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
