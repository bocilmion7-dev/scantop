import streamlit as st

def preprocess_input(text):
    lines = text.strip().split("\n")
    valid = [line.strip() for line in lines if line.strip().isdigit() and len(line.strip()) == 4]
    return [list(map(int, list(item))) for item in valid]

def show_prediction_results(predictions, confidences):
    posisi = ['Ribuan', 'Ratusan', 'Puluhan', 'Satuan']
    st.subheader("🎯 Hasil Prediksi Top-3 Tiap Posisi dengan Confidence")
    for idx, pos in enumerate(posisi):
        top3 = predictions[idx]
        conf = confidences[idx]
        st.markdown(f"**{pos}**:")
        for i in range(3):
            st.write(f"{i+1}. Angka: {top3[i]} | Confidence: {conf[i]:.2%}")

def parse_manual_input(text):
    """
    Mengubah teks manual menjadi list angka 4D.
    Input format: baris per baris, tiap baris 4 digit (contoh: 1234)
    Output: list of list of int, misalnya [[1,2,3,4], [4,5,6,7]]
    """
    lines = text.strip().splitlines()
    data = []
    for line in lines:
        if len(line.strip()) != 4 or not line.strip().isdigit():
            raise ValueError(f"Format salah: {line}")
        data.append([int(d) for d in line.strip()])
    return data
