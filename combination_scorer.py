
import numpy as np
from itertools import product

def score_combinations(predictions, confidences, top_k=10):
    all_combos = list(product(*predictions))
    scores = []
    for combo in all_combos:
        score = np.prod([confidences[i][predictions[i].index(combo[i])] if combo[i] in predictions[i] else 1e-4 for i in range(4)])
        scores.append((combo, score))
    top_combos = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [list(combo) for combo, _ in top_combos]
