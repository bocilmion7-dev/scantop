import numpy as np
from models import build_lstm_block, build_transformer_block, window_data
from markov_model import top6_markov_hybrid
from automl import find_optimal_window

def final_prediction_pipeline(data):
    result_preds = []
    result_confs = []

    for pos in range(4):
        ws = find_optimal_window(data, pos, (10, 30))
        X, y = window_data(data, pos, ws)
        if len(X) < 5:
            raise ValueError(f"Data terlalu sedikit untuk posisi {pos}")
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train LSTM & Transformer
        lstm = build_lstm_block((ws,1))
        trf = build_transformer_block((ws,1))
        lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        trf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        lstm.fit(X, y, epochs=5, verbose=0)
        trf.fit(X, y, epochs=5, verbose=0)

        last_input = X[-1].reshape(1, ws, 1)
        pred_lstm = lstm.predict(last_input, verbose=0)[0]
        pred_trf = trf.predict(last_input, verbose=0)[0]
        pred_nn = (pred_lstm + pred_trf) / 2

        # Markov Hybrid
        top_markov, markov_conf = top6_markov_hybrid(data, pos)
        markov_conf = markov_conf / markov_conf.sum()

        # Final Ensemble
        final_conf = (pred_nn + markov_conf) / 2
        final_conf = final_conf / final_conf.sum()

        top3 = np.argsort(final_conf)[-3:][::-1]
        result_preds.append(top3.tolist())
        result_confs.append(final_conf[top3].tolist())

    return result_preds, result_confs