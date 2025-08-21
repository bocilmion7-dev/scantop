import numpy as np

class MetaLearner:
    def __init__(self):
        pass  # No model needed

    def predict_top3(self, input_features):
        preds, confs = [], []
        for feats in input_features:
            # feats: [LSTM(10), TRF(10), Markov(10)]
            lstm = feats[:10]
            trf = feats[10:20]
            mkv = feats[20:]
            combined = 0.4 * lstm + 0.4 * trf + 0.2 * mkv
            top3 = np.argsort(combined)[-3:][::-1]
            preds.append(top3.tolist())
            confs.append(combined[top3].tolist())
        return preds, confs
