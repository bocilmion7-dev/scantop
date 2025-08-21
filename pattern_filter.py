
from itertools import product

def filter_top_combinations(predictions, top_k=10):
    all_combos = list(product(*predictions))
    filtered = [c for c in all_combos if len(set(c)) == len(c)]  # Rule: semua digit unik
    return filtered[:top_k]
