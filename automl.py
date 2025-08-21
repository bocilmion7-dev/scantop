import numpy as np

def find_optimal_window(data, pos, window_range=(10, 30)):
    best_score = 0
    best_ws = 10
    for ws in range(window_range[0], window_range[1]+1):
        X, y = [], []
        for i in range(len(data) - ws):
            seq = [d[pos] for d in data[i:i+ws]]
            target = data[i+ws][pos]
            X.append(seq)
            y.append(target)
        if len(X) < 5: continue
        X = np.array(X).reshape((-1, ws, 1))
        y = np.array(y)

        last_input = X[-1].reshape(1, ws, 1)
        avg = np.bincount(y, minlength=10) / len(y)
        top3 = np.argsort(avg)[-3:]
        score = avg[top3].sum()

        if score > best_score:
            best_score = score
            best_ws = ws

    return best_ws
