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
# Ganti nama file ini jika Anda menamainya berbeda saat menyimpan
MODEL_FILENAME = 'model_final.pkl'
SCALER_FILENAME = 'scaler_final.pkl'

# FEATURE_ORDER harus sama persis dengan urutan fitur saat model dilatih.
# Sebaiknya salin daftar ini dari output skrip pelatihan Anda.
FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]

# Kolom numerik yang akan di-scaling (harus cocok dengan saat scaler dilatih)
NUMERIC_FEATURES = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins']

# Nama kolom input yang diharapkan dari file yang diunggah
INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

# Opsi untuk fitur kategorikal (one-hot encoding)
BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "unencrypted"]

# ======================================================================================
# FUNGSI-FUNGSI PEMUAT (LOADER)
# ======================================================================================
# Menggunakan cache agar model & scaler hanya dimuat sekali
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        st.error(f"File scaler '{scaler_path}' tidak ditemukan! Harap simpan dari skrip pelatihan Anda.")
        return None
    return joblib.load(scaler_path)

# ======================================================================================
# FUNGSI PRA-PEMROSESAN DATA (FINAL)
# ======================================================================================
def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()
    
    # Pastikan semua kolom input yang dibutuhkan ada
    # Jika tidak ada, isi dengan nilai default agar tidak error
    required_cols = NUMERIC_FEATURES + [INPUT_COL_UNUSUAL_TIME, INPUT_COL_BROWSER, INPUT_COL_PROTOCOL, INPUT_COL_ENCRYPTION]
    for col in required_cols:
        if col not in df.columns:
            if col in NUMERIC_FEATURES:
                df[col] = 0
            else:
                df[col] = "Unknown" 

    # 1. Pisahkan dan Scale data numerik
    numeric_data = df[NUMERIC_FEATURES]
    scaled_numeric_data = scaler.transform(numeric_data)
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=NUMERIC_FEATURES, index=df.index)

    # 2. Proses fitur kategorikal
    categorical_features = {}
    categorical_features['unusual_time_access'] = df[INPUT_COL_UNUSUAL_TIME].apply(lambda x: 1 if x == "Ya" else 0)
    
    for option in BROWSER_OPTIONS:
        categorical_features[f'browser_type_{option}'] = df[INPUT_COL_BROWSER].apply(lambda x: 1 if x == option else 0)
    
    for option in PROTOCOL_OPTIONS:
        categorical_features[f'protocol_type_{option}'] = df[INPUT_COL_PROTOCOL].apply(lambda x: 1 if x == option else 0)

    encryption_mapped = df[INPUT_COL_ENCRYPTION].apply(lambda x: "unencrypted" if x == "None" else x)
    for option in ENCRYPTION_OPTIONS:
        categorical_features[f"encryption_used_{option}"] = encryption_mapped.apply(lambda x: 1 if x == option else 0)

    df_categorical_processed = pd.DataFrame(categorical_features, index=df.index)

    # 3. Gabungkan kembali fitur numerik (sudah di-scale) dan kategorikal
    df_combined = pd.concat([df_scaled, df_categorical_processed], axis=1)

    # 4. Pastikan urutan final benar
    df_processed = df_combined.reindex(columns=FEATURE_ORDER, fill_value=0)
    
    return df_processed

# ======================================================================================
# FUNGSI PREDIKSI
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed):
    if df_processed.empty: return pd.DataFrame()
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)
    df_hasil = pd.DataFrame({
        'Status Deteksi': ['Terancam' if p == 1 else 'Aman' for p in predictions],
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    return df_hasil

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Dashboard Intrusion Detection System (IDS)")

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset (CSV atau Excel)", type=["csv", "xlsx"])
st.sidebar.markdown("---")
st.sidebar.subheader("Panduan Kolom")
st.sidebar.info(
    """
    Pastikan file Anda memiliki kolom dengan nama persis seperti ini:
    - `login_attempts`
    - `failed_logins`
    - `ip_reputation_score`
    - `network_packet_size`
    - `unusual_time_access` (Isi: 'Ya' atau 'Tidak')
    - `browser_type` (Isi: 'Chrome', 'Firefox', dll.)
    - `protocol_type` (Isi: 'TCP', 'UDP', 'ICMP')
    - `encryption_used` (Isi: 'AES', 'DES', atau 'None')
    
    Untuk file CSV, gunakan pemisah titik koma (`;`).
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Capstone Project - Tim Anda")


# --- Muat Model dan Scaler ---
model = load_model(MODEL_FILENAME)
scaler = load_scaler(SCALER_FILENAME)

# --- Tampilan Utama ---
if not model or not scaler:
    st.error("Gagal memuat file `model_terbaik.pkl` dan/atau `scaler.pkl`. Pastikan file-file tersebut ada di folder yang sama dengan `app.py`.")
else:
    if uploaded_file is None:
        st.info("Silakan unggah file dataset melalui panel di sebelah kiri untuk memulai analisis.")
    else:
        try:
            df_input_original = pd.read_csv(uploaded_file, sep=';') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            st.subheader("📄 Pratinjau Data yang Diunggah")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data, mohon tunggu..."):
                    # Proses data dan lakukan prediksi
                    df_processed_for_model = preprocess_dataframe(df_input_original, scaler)
                    df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)

                    if not df_prediksi_hasil.empty:
                        df_tampilan_akhir = pd.concat([df_input_original.reset_index(drop=True), df_prediksi_hasil.reset_index(drop=True)], axis=1)
                        
                        st.markdown("---")
                        st.subheader("📊 Hasil Analisis & Visualisasi")

                        # --- Metrik Utama ---
                        total_data = len(df_tampilan_akhir)
                        jumlah_ancaman = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                        jumlah_aman = total_data - jumlah_ancaman

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Data Dianalisis", f"{total_data}")
                        with col2:
                            st.metric("🚨 Terdeteksi Ancaman", f"{jumlah_ancaman}")
                        with col3:
                            st.metric("✅ Terdeteksi Aman", f"{jumlah_aman}")

                        st.markdown("<br>", unsafe_allow_html=True)

                        # --- Visualisasi ---
                        col_viz1, col_viz2 = st.columns(2)

                        with col_viz1:
                            # Pie Chart Proporsi Status
                            st.write("**Proporsi Status Deteksi**")
                            if total_data > 0:
                                status_counts = df_tampilan_akhir['Status Deteksi'].value_counts()
                                fig1, ax1 = plt.subplots(figsize=(5, 4))
                                ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FF4B4B', '#3DDC97'])
                                ax1.axis('equal') # Memastikan pie chart berbentuk lingkaran.
                                st.pyplot(fig1)
                            else:
                                st.write("Tidak ada data untuk ditampilkan.")

                        with col_viz2:
                            # Bar Chart Ancaman per Browser
                            st.write("**Ancaman Berdasarkan Tipe Browser**")
                            if jumlah_ancaman > 0:
                                df_ancaman = df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam']
                                ancaman_per_browser = df_ancaman[INPUT_COL_BROWSER].value_counts()
                                st.bar_chart(ancaman_per_browser)
                            else:
                                st.write("Tidak ada data ancaman yang terdeteksi.")

                        # --- Distribusi Fitur ---
                        st.write("**Distribusi Fitur Penting: Skor Reputasi IP**")
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        sns.histplot(data=df_tampilan_akhir, x=INPUT_COL_IP_REPUTATION, hue='Status Deteksi', kde=True, palette={'Aman': '#3DDC97', 'Terancam': '#FF4B4B'}, ax=ax2)
                        st.pyplot(fig2)

                        # --- Tabel Hasil Lengkap ---
                        st.markdown("---")
                        st.subheader("📋 Tabel Hasil Analisis Lengkap")
                        st.dataframe(df_tampilan_akhir, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.exception(e)

