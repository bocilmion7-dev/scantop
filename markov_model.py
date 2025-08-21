import numpy as np
from collections import defaultdict

def top6_markov_order1(data, pos):
    trans = defaultdict(lambda: np.zeros(10))
    for i in range(1, len(data)):
        prev = data[i-1][pos]
        curr = data[i][pos]
        trans[prev][curr] += 1
    for k in trans:
        trans[k] /= trans[k].sum()
    last_digit = data[-1][pos]
    probs = trans[last_digit] if last_digit in trans else np.ones(10)/10
    top6 = np.argsort(probs)[-6:][::-1]
    return top6, probs

def top6_markov_order2(data, pos):
    trans = defaultdict(lambda: np.zeros(10))
    for i in range(2, len(data)):
        prev = (data[i-2][pos], data[i-1][pos])
        curr = data[i][pos]
        trans[prev][curr] += 1
    for k in trans:
        trans[k] /= trans[k].sum()
    key = (data[-2][pos], data[-1][pos])
    probs = trans[key] if key in trans else np.ones(10)/10
    top6 = np.argsort(probs)[-6:][::-1]
    return top6, probs

def top6_markov_hybrid(data, pos):
    top1, p1 = top6_markov_order1(data, pos)
    top2, p2 = top6_markov_order2(data, pos)
    combined = (p1 + p2) / 2
    top = np.argsort(combined)[-6:][::-1]
    return top, combined