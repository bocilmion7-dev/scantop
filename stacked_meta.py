
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MetaLearner:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_top3(self, meta_inputs):
        if not self.is_fitted:
            raise Exception("Meta model belum dilatih")
        top3_preds, top3_confs = [], []
        for x in meta_inputs:
            pred_probs = self.model.predict_proba([x])[0]
            top3 = np.argsort(pred_probs)[-3:][::-1]
            top3_preds.append(top3.tolist())
            top3_confs.append(pred_probs[top3].tolist())
        return top3_preds, top3_confs
