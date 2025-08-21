
import numpy as np
from collections import Counter

def delay_digit(data, pos):
    delay = [0] * 10
    last_seen = [-1] * 10
    for i, row in enumerate(data):
        digit = row[pos]
        for d in range(10):
            if last_seen[d] != -1:
                delay[d] = i - last_seen[d]
        last_seen[digit] = i
    return delay

def digit_frequencies(data, pos):
    count = Counter(row[pos] for row in data)
    freqs = np.zeros(10)
    for k, v in count.items():
        freqs[k] = v
    return freqs / freqs.sum()

def compute_pattern_score(data, pos):
    freqs = digit_frequencies(data, pos)
    delays = delay_digit(data, pos)
    delay_scores = np.array([1 / (d + 1) for d in delays])
    score = 0.7 * freqs + 0.3 * delay_scores
    return score / score.sum()
