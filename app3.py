import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns

# ======================================================================================
# KONFIGURASI PUSAT
# ======================================================================================
MODEL_FILENAME = 'best_random_forest_model (1).pkl'

# Daftar fitur yang diharapkan oleh model (TANPA awalan 'num__' atau 'cat__')
FEATURE_ORDER = [
    'login_attempts', 'failed_logins', 'ip_reputation_score', 'network_packet_size',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Firefox',
    'browser_type_Edge', 'browser_type_Safari', 'browser_type_Unknown',
    'protocol_type_TCP', 'protocol_type_UDP', 'protocol_type_ICMP',
    'encryption_used_AES', 'encryption_used_DES', 'encryption_used_None'
]

# Nama kolom ini HARUS SESUAI dengan nama kolom di file CSV/Excel yang diunggah.
# Sesuaikan string di bawah ini jika nama kolom di file Anda berbeda.
INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

# Opsi untuk fitur kategorikal
BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "None"]

# --- DEBUG FLAG ---
DEBUG_MODE = False # Set ke True untuk melihat output debugging

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
        # Verifikasi jumlah fitur yang diharapkan model
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(FEATURE_ORDER):
            st.error(f"FATAL: Model mengharapkan {model.n_features_in_} fitur, "
                     f"tetapi FEATURE_ORDER memiliki {len(FEATURE_ORDER)} fitur.")
            return None
        if DEBUG_MODE:
            st.success(f"Model '{model_path}' berhasil dimuat.")
            if hasattr(model, 'n_features_in_'):
                   st.write(f"Model mengharapkan {model.n_features_in_} fitur.")
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# ======================================================================================
# FUNGSI UNTUK PRA-PEMROSESAN DATASET
# ======================================================================================
def preprocess_dataframe(df_input_original):
    if DEBUG_MODE:
        st.subheader("--- DEBUG: PREPROCESS ---")
        st.write("Memulai `preprocess_dataframe`...")
        st.write("Kolom di df_input_original:", df_input_original.columns.tolist())
        st.dataframe(df_input_original.head())

    processed_rows = []
    for i, row in df_input_original.iterrows():
        final_model_inputs = {}

        # 1. Fitur Numerik Langsung (tanpa awalan)
        final_model_inputs['login_attempts'] = row.get(INPUT_COL_LOGIN_ATTEMPTS, 0)
        final_model_inputs['failed_logins'] = row.get(INPUT_COL_FAILED_LOGINS, 0)
        final_model_inputs['ip_reputation_score'] = row.get(INPUT_COL_IP_REPUTATION, 0)
        final_model_inputs['network_packet_size'] = row.get(INPUT_COL_PACKET_SIZE, 0)

        # 2. Fitur Kategorikal - One-Hot Encoding Manual
        # unusual_time_access
        user_choice_unusual_time = row.get(INPUT_COL_UNUSUAL_TIME, "Tidak")
        final_model_inputs['unusual_time_access'] = 1 if user_choice_unusual_time == "Ya" else 0

        # browser_type
        user_choice_browser = row.get(INPUT_COL_BROWSER, "Unknown")
        for browser_option in BROWSER_OPTIONS:
            feature_name_encoded = f"browser_type_{browser_option}" # Nama fitur tanpa 'cat__'
            final_model_inputs[feature_name_encoded] = 1 if user_choice_browser == browser_option else 0

        # protocol_type
        user_choice_protocol = row.get(INPUT_COL_PROTOCOL, "TCP")
        for protocol_option in PROTOCOL_OPTIONS:
            feature_name_encoded = f"protocol_type_{protocol_option}" # Nama fitur tanpa 'cat__'
            final_model_inputs[feature_name_encoded] = 1 if user_choice_protocol == protocol_option else 0

        # encryption_used
        user_choice_encryption = row.get(INPUT_COL_ENCRYPTION, "None")
        for encryption_option in ENCRYPTION_OPTIONS:
            feature_name_encoded = f"encryption_used_{encryption_option}" # Nama fitur tanpa 'cat__'
            final_model_inputs[feature_name_encoded] = 1 if user_choice_encryption == encryption_option else 0

        # Safety net untuk memastikan semua fitur ada
        for feature in FEATURE_ORDER:
            if feature not in final_model_inputs:
                final_model_inputs[feature] = 0

        processed_rows.append(final_model_inputs)

    # Buat DataFrame dengan urutan kolom yang benar sesuai FEATURE_ORDER
    df_processed = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)

    if DEBUG_MODE:
        st.write("`preprocess_dataframe` selesai.")
        st.write("Lima baris pertama `df_processed` (data yang akan masuk ke model):")
        st.dataframe(df_processed.head())
        st.write("Nama kolom yang dikirim ke model:", df_processed.columns.tolist())
        st.subheader("--- AKHIR DEBUG: PREPROCESS ---")

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

if DEBUG_MODE:
    st.warning("MODE DEBUG AKTIF. Output tambahan akan ditampilkan.")

model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV (pemisah ';') atau Excel", type=["csv", "xlsx", "xls"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("PENTING: Nama Kolom di File Anda")
    # Teks di sidebar diperbarui untuk mencerminkan nama kolom yang benar
    st.sidebar.info(
        f"""
        Pastikan dataset Anda memiliki kolom dengan nama yang **PERSIS SAMA** seperti ini:
        - `{INPUT_COL_LOGIN_ATTEMPTS}` (Numerik)
        - `{INPUT_COL_FAILED_LOGINS}` (Numerik)
        - `{INPUT_COL_IP_REPUTATION}` (Numerik)
        - `{INPUT_COL_PACKET_SIZE}` (Numerik)
        - `{INPUT_COL_UNUSUAL_TIME}` (Teks: 'Ya' atau 'Tidak')
        - `{INPUT_COL_BROWSER}` (Teks: e.g., 'Chrome', 'Firefox')
        - `{INPUT_COL_PROTOCOL}` (Teks: e.g., 'TCP', 'UDP')
        - `{INPUT_COL_ENCRYPTION}` (Teks: e.g., 'AES', 'None')

        Untuk file CSV, pastikan pemisah kolom adalah titik koma (`;`).
        Kesalahan nama kolom akan menyebabkan kegagalan atau hasil tidak akurat.
        """
    )

    if uploaded_file is not None:
        try:
            df_input_original = None
            if uploaded_file.name.endswith('.csv'):
                df_input_original = pd.read_csv(uploaded_file, sep=';')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df_input_original = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
                st.stop()

            st.subheader("📄 Data Awal yang Diunggah (Contoh 5 Baris Pertama):")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis pada Seluruh Dataset", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data... ⏳"):
                    # Pra-pemrosesan data
                    df_processed_for_model = preprocess_dataframe(df_input_original.copy())

                    if df_processed_for_model.empty:
                        st.warning("Tidak ada data yang bisa diproses.")
                    else:
                        # Lakukan prediksi
                        df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)

                        if not df_prediksi_hasil.empty:
                            # Gabungkan data asli dengan hasil prediksi
                            df_tampilan_akhir = pd.concat([
                                df_input_original.reset_index(drop=True),
                                df_prediksi_hasil.reset_index(drop=True)
                            ], axis=1)

                            st.subheader("📊 Hasil Analisis Lengkap per Data:")
                            kolom_tampil_final = list(df_input_original.columns) + ['Status Deteksi', 'Probabilitas Ancaman (%)']
                            st.dataframe(df_tampilan_akhir[kolom_tampil_final], use_container_width=True)
                            st.markdown("---")

                            # Tampilkan ringkasan dan statistik
                            st.subheader("📈 Ringkasan dan Statistik Hasil Analisis:")
                            jumlah_total_data = len(df_tampilan_akhir)
                            jumlah_terancam = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                            jumlah_aman = jumlah_total_data - jumlah_terancam
                            persentase_terancam = (jumlah_terancam / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0
                            persentase_aman = 100 - persentase_terancam

                            col1, col2, col3 = st.columns(3)
                            col1.metric(label="Total Data Dianalisis", value=f"{jumlah_total_data} baris")
                            col2.metric(label="🚨 Data Terdeteksi Ancaman", value=f"{jumlah_terancam} baris",
                                        delta=f"{persentase_terancam:.2f}% dari total", delta_color="inverse")
                            col3.metric(label="✅ Data Terdeteksi Aman", value=f"{jumlah_aman} baris",
                                        delta=f"{persentase_aman:.2f}% dari total", delta_color="normal")

                            # Visualisasi hasil
                            if jumlah_total_data > 0:
                                st.subheader("Visualisasi Hasil:")
                                # Bar chart
                                data_grafik_status_df = pd.DataFrame({
                                    'Status': ['Terancam', 'Aman'],
                                    'Jumlah': [jumlah_terancam, jumlah_aman]
                                })
                                st.bar_chart(data_grafik_status_df.set_index('Status'), color=["#FF4B4B", "#3DDC97"])

                                # Histogram/Distplot
                                if jumlah_terancam > 0:
                                    st.write("Distribusi Probabilitas Ancaman (%) untuk Data Terancam:")
                                    data_terancam_probs = df_tampilan_akhir[
                                        df_tampilan_akhir['Status Deteksi'] == 'Terancam'
                                    ]['Probabilitas Ancaman (%)']
                                    fig, ax = plt.subplots()
                                    sns.histplot(data_terancam_probs, kde=True, color="#FF4B4B", ax=ax)
                                    ax.set_xlabel("Probabilitas Ancaman (%)")
                                    ax.set_ylabel("Jumlah")
                                    ax.set_title('Distribusi Probabilitas Ancaman (Data Terancam)')
                                    st.pyplot(fig)

        except pd.errors.ParserError:
            st.error("Gagal mem-parsing file. Pastikan format CSV benar dan pemisah kolom adalah titik koma (';').")
        except KeyError as e:
            st.error(f"Kolom yang dibutuhkan tidak ditemukan: {e}. "
                     "Periksa kembali nama kolom di file Anda dan pastikan sesuai dengan yang diharapkan di sidebar.")
        except Exception as e:
            st.error(f"Terjadi kesalahan umum: {e}")
            st.exception(e) # Menampilkan traceback untuk debugging
    else:
        st.info("Silakan unggah dataset (CSV atau Excel) melalui panel kiri untuk memulai analisis.")
else:
    st.error("Model tidak dapat dimuat. Aplikasi tidak dapat melanjutkan.")

st.markdown("---")
st.caption("Dashboard IDS | Dibuat dengan Streamlit & Scikit-learn")
