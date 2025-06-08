import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================================================================
# KONFIGURASI PUSAT
# ======================================================================================
MODEL_FILENAME = 'best_random_forest_model (1).pkl'

# DAFTAR FITUR YANG DIPERBAIKI: Menggunakan 'unencrypted'
FEATURE_ORDER = [
    'login_attempts', 'failed_logins', 'ip_reputation_score',
    'network_packet_size', 'unusual_time_access', 'browser_type_Chrome',
    'browser_type_Firefox', 'browser_type_Edge', 'browser_type_Safari',
    'browser_type_Unknown', 'protocol_type_TCP', 'protocol_type_UDP',
    'protocol_type_ICMP', 'encryption_used_AES', 'encryption_used_DES',
    'encryption_used_unencrypted'  # <- PERBAIKAN 1
]

# NAMA KOLOM INPUT: Dibiarkan sederhana
INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

# Opsi untuk fitur kategorikal (DIPERBAIKI)
BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "unencrypted"] # <- PERBAIKAN 2

# --- DEBUG FLAG ---
DEBUG_MODE = False

# ======================================================================================
# FUNGSI UNTUK MEMUAT MODEL
# ======================================================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# ======================================================================================
# FUNGSI UNTUK PRA-PEMROSESAN DATASET (DIPERBAIKI)
# ======================================================================================
def preprocess_dataframe(df_input):
    processed_rows = []
    for i, row in df_input.iterrows():
        final_model_inputs = {}

        # 1. Fitur Numerik
        final_model_inputs['login_attempts'] = row.get(INPUT_COL_LOGIN_ATTEMPTS, 0)
        final_model_inputs['failed_logins'] = row.get(INPUT_COL_FAILED_LOGINS, 0)
        final_model_inputs['ip_reputation_score'] = row.get(INPUT_COL_IP_REPUTATION, 0)
        final_model_inputs['network_packet_size'] = row.get(INPUT_COL_PACKET_SIZE, 0)

        # 2. Fitur Kategorikal
        final_model_inputs['unusual_time_access'] = 1 if row.get(INPUT_COL_UNUSUAL_TIME, "Tidak") == "Ya" else 0
        
        user_choice_browser = row.get(INPUT_COL_BROWSER, "Unknown")
        for option in BROWSER_OPTIONS:
            final_model_inputs[f"browser_type_{option}"] = 1 if user_choice_browser == option else 0
        
        user_choice_protocol = row.get(INPUT_COL_PROTOCOL, "TCP")
        for option in PROTOCOL_OPTIONS:
            final_model_inputs[f"protocol_type_{option}"] = 1 if user_choice_protocol == option else 0

        # encryption_used (DENGAN LOGIKA PEMETAAN - PERBAIKAN 3)
        user_choice_encryption = row.get(INPUT_COL_ENCRYPTION, "None")
        if user_choice_encryption == "None":
            user_choice_encryption = "unencrypted"
        
        for option in ENCRYPTION_OPTIONS:
            final_model_inputs[f"encryption_used_{option}"] = 1 if user_choice_encryption == option else 0
        
        processed_rows.append(final_model_inputs)

    df_processed = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)
    df_processed.fillna(0, inplace=True)
    return df_processed


# ======================================================================================
# FUNGSI UNTUK PREDIKSI PADA DATASET
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed_for_predict):
    if df_processed_for_predict.empty:
        return pd.DataFrame()
    predictions = model.predict(df_processed_for_predict)
    prediction_probas = model.predict_proba(df_processed_for_predict)
    df_hasil = pd.DataFrame({
        'Status Deteksi Numerik': predictions,
        'Probabilitas Normal (%)': (prediction_probas[:, 0] * 100).round(2),
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    df_hasil['Status Deteksi'] = df_hasil['Status Deteksi Numerik'].apply(lambda x: 'Terancam' if x == 1 else 'Aman')
    return df_hasil

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard - Analisis Dataset")

model = load_model(MODEL_FILENAME)
if model:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV (pemisah ';') atau Excel", type=["csv", "xlsx", "xls"])
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("PENTING: Nama Kolom di File Anda")
    st.sidebar.info(
        f"""
        Pastikan dataset Anda memiliki kolom dengan nama yang **PERSIS SAMA** seperti ini:
        - `{INPUT_COL_LOGIN_ATTEMPTS}`
        - `{INPUT_COL_FAILED_LOGINS}`
        - `{INPUT_COL_IP_REPUTATION}`
        - `{INPUT_COL_PACKET_SIZE}`
        - `{INPUT_COL_UNUSUAL_TIME}` (Isi: 'Ya' atau 'Tidak')
        - `{INPUT_COL_BROWSER}` (Isi: misal, 'Chrome', 'Firefox')
        - `{INPUT_COL_PROTOCOL}` (Isi: misal, 'TCP', 'UDP')
        - `{INPUT_COL_ENCRYPTION}` (Isi: 'AES', 'DES', atau **'None'**)
        """
    )

    if uploaded_file is not None:
        try:
            df_input_original = pd.read_csv(uploaded_file, sep=';') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            st.subheader("📄 Data Awal yang Diunggah (Contoh 5 Baris Pertama):")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis pada Seluruh Dataset", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data... ⏳"):
                    df_processed_for_model = preprocess_dataframe(df_input_original.copy())
                    df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)

                    if not df_prediksi_hasil.empty:
                        df_tampilan_akhir = pd.concat([df_input_original.reset_index(drop=True), df_prediksi_hasil.reset_index(drop=True)], axis=1)
                        st.subheader("📊 Hasil Analisis Lengkap per Data:")
                        kolom_tampil_final = list(df_input_original.columns) + ['Status Deteksi', 'Probabilitas Ancaman (%)']
                        st.dataframe(df_tampilan_akhir[kolom_tampil_final], use_container_width=True)
                        st.markdown("---")

                        st.subheader("📈 Ringkasan dan Statistik Hasil Analisis:")
                        jumlah_total_data = len(df_tampilan_akhir)
                        jumlah_terancam = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                        jumlah_aman = jumlah_total_data - jumlah_terancam
                        persentase_terancam = (jumlah_terancam / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0
                        persentase_aman = 100 - persentase_terancam

                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Total Data Dianalisis", value=f"{jumlah_total_data} baris")
                        col2.metric(label="🚨 Data Terdeteksi Ancaman", value=f"{jumlah_terancam} baris", delta=f"{persentase_terancam:.2f}%", delta_color="inverse")
                        col3.metric(label="✅ Data Terdeteksi Aman", value=f"{jumlah_aman} baris", delta=f"{persentase_aman:.2f}%", delta_color="normal")

                        if jumlah_total_data > 0:
                            st.subheader("Visualisasi Hasil:")
                            data_grafik_status_df = pd.DataFrame({'Terancam': [jumlah_terancam], 'Aman': [jumlah_aman]})
                            st.bar_chart(data_grafik_status_df, color=["#FF4B4B", "#3DDC97"])

                            if jumlah_terancam > 0:
                                st.write("Distribusi Probabilitas Ancaman (%) untuk Data Terancam:")
                                data_terancam_probs = df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam']['Probabilitas Ancaman (%)']
                                fig, ax = plt.subplots()
                                sns.histplot(data_terancam_probs, kde=True, color="#FF4B4B", ax=ax)
                                ax.set_xlabel("Probabilitas Ancaman (%)")
                                ax.set_ylabel("Jumlah")
                                st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.exception(e)
    else:
        st.info("Silakan unggah dataset (CSV atau Excel) melalui panel kiri untuk memulai analisis.")
else:
    st.error("Model tidak dapat dimuat. Pastikan file " + MODEL_FILENAME + " ada di folder yang sama.")

st.markdown("---")
st.caption("Dashboard IDS | Dibuat dengan Streamlit")
