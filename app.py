import streamlit as st
from ensemble_stack_v3 import final_prediction_pipeline
from utils import parse_manual_input
from visualizer import plot_confidences, plot_top_combinations
import pandas as pd

st.set_page_config(page_title="Prediksi Angka 4D SOTA", layout="wide")
st.title("🧠 Prediksi Angka 4D - Full Keras Mode")

st.markdown("### Masukkan Data Historis 4D (contoh format 1234 per baris):")
manual_input = st.text_area("Input Data Manual:", height=200)

if st.button("🚀 Jalankan Prediksi"):
    try:
        data = parse_manual_input(manual_input)
        hasil = final_prediction_pipeline(data)
    except Exception as e:
        st.error(f"❌ Gagal memproses: {e}")
        st.stop()

    st.success("✅ Prediksi sukses dengan model keras!")

    posisi_label = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
    st.markdown("### 🎯 Top-3 Prediksi per Posisi:")
    for i in range(4):
        st.markdown(f"**{posisi_label[i]}**")
        plot_confidences(hasil["top3_per_posisi"][i], hasil["confidences"][i], key=f"keras_{i}")

    st.markdown("### 💡 Top-10 Kombinasi 4D Potensial:")
    df_top = pd.DataFrame(hasil["top10_kombinasi"], columns=["Angka 4D", "Skor"])
    st.dataframe(df_top, use_container_width=True)
    plot_top_combinations(df_top)

st.markdown("---")
st.caption("🔁 Mode Keras Stabil: LSTM + Transformer (epochs rendah, clear session aktif)")
