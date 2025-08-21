import numpy as np
import streamlit as st

def simulate_prediction_accuracy(data, prediction_fn, true_data_len=10):
    """
    Simulasikan akurasi dengan mengambil prediksi top-3 terhadap 10 data terakhir.
    """
    if len(data) < true_data_len + 20:
        st.warning("Tidak cukup data untuk simulasi akurasi.")
        return None

    X_data = data[:-true_data_len]
    real_targets = data[-true_data_len:]
    top1_hits, top3_hits = 0, 0

    for i in range(true_data_len):
        current_data = X_data + real_targets[:i]
        pred, _ = prediction_fn(current_data)

        for j in range(4):
            top3 = pred[j]
            true_digit = real_targets[i][j]
            if true_digit == top3[0]:
                top1_hits += 1
            if true_digit in top3:
                top3_hits += 1

    total = true_data_len * 4
    top1_acc = top1_hits / total
    top3_acc = top3_hits / total
    return top1_acc, top3_acc
