from sklearn.linear_model import LogisticRegression


class CreditScoreRegressionClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifier = LogisticRegression(random_state=random_state)

    def train(self, origination_x):
        X = origination_x[:, 1:]
        y = origination_x[:, 0].astype('int')
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
