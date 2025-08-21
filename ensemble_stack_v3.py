import numpy as np
import tensorflow.keras.backend as K
from models import build_lstm_block, build_transformer_block, window_data
from markov_model import top6_markov_hybrid
from automl import find_optimal_window
from meta_learner import MetaLearner
from pattern_digit import compute_pattern_score
from pattern_filter import filter_top_combinations
from confidence_calibrator import temperature_scale
from combination_scorer import score_combinations
from drift_monitor import log_prediction

import tensorflow.keras.backend as K

def simulate_model_accuracy(X, y, model, last_input):
    K.clear_session()
    try:
        model.fit(X, y, epochs=2, verbose=0)
        preds = model.predict(X, verbose=0)
        top3_hits = sum(y[i] in np.argsort(preds[i])[-3:] for i in range(len(y)))
        last_pred = model.predict(last_input, verbose=0)[0]
        return top3_hits / len(y), last_pred
    except Exception as e:
        print(f"[ERROR keras training]: {e}")
        dummy = np.ones(10) / 10
        return 0.0, dummy
        
def final_prediction_pipeline(data):
    result_preds = []
    result_confs = []
    meta_inputs = []

    for pos in range(4):
        ws = find_optimal_window(data, pos, (10, 30))
        X, y = window_data(data, pos, ws)
        if len(X) < 5:
            raise ValueError(f"Data terlalu sedikit untuk posisi {pos}")
        X = X.reshape((X.shape[0], X.shape[1], 1))
        last_input = X[-1].reshape(1, ws, 1)

        # === Model training
        lstm = build_lstm_block((ws,1))
        trf = build_transformer_block((ws,1))
        lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        trf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        acc_lstm, pred_lstm = simulate_model_accuracy(X, y, lstm, last_input)
        acc_trf, pred_trf = simulate_model_accuracy(X, y, trf, last_input)
        _, pred_markov = top6_markov_hybrid(data, pos)
        pred_markov = pred_markov / pred_markov.sum()

        # === Pattern-aware scoring
        pattern_score = compute_pattern_score(data, pos)
        pred_lstm *= pattern_score
        pred_trf *= pattern_score
        pred_markov *= pattern_score

        # === Confidence calibration
        pred_lstm = temperature_scale(pred_lstm, temperature=1.5)
        pred_trf = temperature_scale(pred_trf, temperature=1.5)
        pred_markov = temperature_scale(pred_markov, temperature=1.5)

        # === Meta learner input
        meta_input = np.concatenate([pred_lstm, pred_trf, pred_markov])
        meta_inputs.append(meta_input)

    # === Meta-learner
    meta = MetaLearner()
    preds, confs = meta.predict_top3(meta_inputs)

    result_preds = preds
    result_confs = confs

    # === Probabilistic scoring + filter
    kombinasi_skored = score_combinations(result_preds, result_confs, top_k=50)
    top10_kombinasi = filter_top_combinations(kombinasi_skored, top_k=10)

    # === Logging real accuracy
    for i in range(4):
        log_prediction(real_digit=data[-1][i], predicted_top3=result_preds[i], position=i)

    return {
        "top3_per_posisi": result_preds,
        "confidences": result_confs,
        "top10_kombinasi": top10_kombinasi
    }
