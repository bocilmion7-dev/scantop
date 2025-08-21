
import numpy as np

def temperature_scale(logits, temperature=1.5):
    logits = np.log(logits + 1e-8)
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / np.sum(exp_scaled)
