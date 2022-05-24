from sklearn.linear_model import LogisticRegression
import json

with open('config/LR_config.json', 'r') as f:
    LR_json = json.load(f)


class LRClassifier:

    def __init__(self,):
        self.LRMachine=LogisticRegression(**LR_json)


    def train(self, train_x, train_y, _sample_weight=None):
        self.LRMachine.fit(train_x,train_y, sample_weight=_sample_weight)


    def predict_proba(self, pred_x):
        return self.LRMachine.predict_proba(pred_x)

