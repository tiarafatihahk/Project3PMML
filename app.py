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
FEATURE_ORDER = [
    'num__login_attempts', 'num__failed_logins', 'num__ip_reputation_score',
    'num__network_packet_size', 'cat__unusual_time_access', 'cat__browser_type_Chrome',
    'cat__browser_type_Firefox', 'cat__browser_type_Edge', 'cat__browser_type_Safari',
    'cat__browser_type_Unknown', 'cat__protocol_type_TCP', 'cat__protocol_type_UDP',
    'cat__protocol_type_ICMP', 'cat__encryption_used_AES', 'cat__encryption_used_DES',
    'cat__encryption_used_None','17','18','19','20'
]
ORIGINAL_NUMERIC_FEATURES = [
    'num__login_attempts', 'num__failed_logins',
    'num__ip_reputation_score', 'num__network_packet_size'
]
# Nama kolom ini HARUS SESUAI dengan nama kolom di file CSV/Excel yang diunggah pengguna
# Contoh: jika di file CSV ada kolom 'jumlah_login', maka di sini harus 'jumlah_login'
# dan nanti di preprocess_dataframe, row.get('jumlah_login', 0)
# Untuk konsistensi dengan FEATURE_ORDER, kita asumsikan nama kolom input akan mirip
# Namun, ini adalah poin KRUSIAL untuk diverifikasi.
INPUT_COL_LOGIN_ATTEMPTS = 'num__login_attempts' # SESUAIKAN JIKA NAMA KOLOM DI FILE BERBEDA
INPUT_COL_FAILED_LOGINS = 'num__failed_logins' # SESUAIKAN
INPUT_COL_IP_REPUTATION = 'num__ip_reputation_score' # SESUAIKAN
INPUT_COL_PACKET_SIZE = 'num__network_packet_size' # SESUAIKAN
INPUT_COL_UNUSUAL_TIME = 'cat__unusual_time_access' # SESUAIKAN
INPUT_COL_BROWSER = 'cat__browser_type' # SESUAIKAN
INPUT_COL_PROTOCOL = 'cat__protocol_type' # SESUAIKAN
INPUT_COL_ENCRYPTION = 'cat__encryption_used' # SESUAIKAN

BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "None"]

# --- DEBUG FLAG ---
DEBUG_MODE = False # Set ke False jika sudah tidak debugging

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
def preprocess_dataframe(df_input_original_for_preprocess): # Ganti nama parameter agar jelas
    if DEBUG_MODE:
        st.subheader("--- DEBUG: PREPROCESS ---")
        st.write("Memulai `preprocess_dataframe`...")
        st.write("Kolom di df_input_original_for_preprocess:", df_input_original_for_preprocess.columns.tolist())
        st.write(f"Lima baris pertama df_input_original_for_preprocess:")
        st.dataframe(df_input_original_for_preprocess.head())


    processed_rows = []
    for i, row in df_input_original_for_preprocess.iterrows(): # Tambahkan i untuk melacak baris
        final_model_inputs = {} # Dictionary untuk menyimpan fitur yang sudah diproses untuk satu baris

        # 1. Fitur Numerik Langsung
        # Ambil dari kolom input yang sudah didefinisikan (dan disesuaikan dengan nama kolom di file)
        final_model_inputs['num__login_attempts'] = row.get(INPUT_COL_LOGIN_ATTEMPTS, 0)
        final_model_inputs['num__failed_logins'] = row.get(INPUT_COL_FAILED_LOGINS, 0)
        final_model_inputs['num__ip_reputation_score'] = row.get(INPUT_COL_IP_REPUTATION, 0)
        final_model_inputs['num__network_packet_size'] = row.get(INPUT_COL_PACKET_SIZE, 0)

        # 2. Fitur Kategorikal - Perlu One-Hot Encoding manual
        # cat__unusual_time_access
        user_choice_unusual_time = row.get(INPUT_COL_UNUSUAL_TIME, "Tidak") # Default "Tidak"
        final_model_inputs['cat__unusual_time_access'] = 1 if user_choice_unusual_time == "Ya" else 0

        # cat__browser_type
        user_choice_browser = row.get(INPUT_COL_BROWSER, "Unknown") # Default "Unknown"
        for browser_option in BROWSER_OPTIONS:
            feature_name_encoded = f"cat__browser_type_{browser_option}"
            if feature_name_encoded in FEATURE_ORDER:
                 final_model_inputs[feature_name_encoded] = 1 if user_choice_browser == browser_option else 0

        # cat__protocol_type
        user_choice_protocol = row.get(INPUT_COL_PROTOCOL, "TCP") # Default "TCP" atau yang paling umum
        for protocol_option in PROTOCOL_OPTIONS:
            feature_name_encoded = f"cat__protocol_type_{protocol_option}"
            if feature_name_encoded in FEATURE_ORDER:
                final_model_inputs[feature_name_encoded] = 1 if user_choice_protocol == protocol_option else 0

        # cat__encryption_used
        user_choice_encryption = row.get(INPUT_COL_ENCRYPTION, "None") # Default "None"
        for encryption_option in ENCRYPTION_OPTIONS:
            feature_name_encoded = f"cat__encryption_used_{encryption_option}"
            if feature_name_encoded in FEATURE_ORDER:
                final_model_inputs[feature_name_encoded] = 1 if user_choice_encryption == encryption_option else 0
        
        # Safety net: Pastikan semua fitur di FEATURE_ORDER ada di final_model_inputs
        # Jika ada fitur di FEATURE_ORDER yang belum terdefinisi di atas, set ke 0 (atau nilai default lain)
        for feature_in_order in FEATURE_ORDER:
            if feature_in_order not in final_model_inputs:
                if DEBUG_MODE and i < 1: # Cetak hanya untuk baris pertama jika ada fitur yang "hilang"
                    st.warning(f"DEBUG: Fitur '{feature_in_order}' dari FEATURE_ORDER tidak ada di final_model_inputs setelah pemrosesan fitur input. Di-set ke 0.")
                final_model_inputs[feature_in_order] = 0
        
        # DEBUG: Cetak final_model_inputs untuk beberapa baris pertama
        if DEBUG_MODE and i < 3: # Ubah angka 3 sesuai kebutuhan
            st.write(f"DEBUG: `final_model_inputs` untuk baris input ke-{i}:")
            # Tampilkan dalam bentuk yang lebih mudah dibaca
            st.json(final_model_inputs) # st.json lebih rapi untuk dictionary

        processed_rows.append(final_model_inputs)

    # Buat DataFrame dari list dictionary, pastikan urutan kolom sesuai FEATURE_ORDER
    df_processed = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)
    
    if DEBUG_MODE:
        st.write("`preprocess_dataframe` selesai.")
        st.write("Lima baris pertama `df_processed` (data yang akan masuk ke model):")
        st.dataframe(df_processed.head())
        st.write("Statistik deskriptif `df_processed`:")
        st.dataframe(df_processed.describe())
        st.subheader("--- AKHIR DEBUG: PREPROCESS ---")

    return df_processed

# ======================================================================================
# FUNGSI UNTUK PREDIKSI PADA DATASET
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed_for_predict): # Ganti nama parameter
    if DEBUG_MODE:
        st.subheader("--- DEBUG: PREDICT ---")
        st.write("Memulai `model_prediksi_ancaman_dataset`...")
        st.write("Lima baris pertama `df_processed_for_predict` (data yang diterima fungsi prediksi):")
        st.dataframe(df_processed_for_predict.head())

    if df_processed_for_predict.empty:
        if DEBUG_MODE: st.warning("DEBUG: `df_processed_for_predict` kosong, prediksi dibatalkan.")
        return pd.DataFrame()

    predictions = model.predict(df_processed_for_predict)
    prediction_probas = model.predict_proba(df_processed_for_predict)

    if DEBUG_MODE:
        st.write("DEBUG: `predictions` (raw dari model):")
        st.write(predictions[:5]) # Tampilkan 5 prediksi pertama
        st.write("DEBUG: `prediction_probas` (raw dari model):")
        st.write(prediction_probas[:5]) # Tampilkan 5 probabilitas pertama

    df_hasil = pd.DataFrame({
        'Status Deteksi Numerik': predictions,
        'Probabilitas Normal (%)': (prediction_probas[:, 0] * 100).round(2),
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    df_hasil['Status Deteksi'] = df_hasil['Status Deteksi Numerik'].apply(lambda x: 'Terancam' if x == 1 else 'Aman')
    # Kolom 'Tingkat Kepercayaan Ancaman (%)' sudah ada dari 'Probabilitas Ancaman (%)'
    # df_hasil['Tingkat Kepercayaan Ancaman (%)'] = df_hasil['Probabilitas Ancaman (%)'] # Ini redundan

    if DEBUG_MODE:
        st.write("DEBUG: `df_hasil` (setelah prediksi dan penambahan kolom):")
        st.dataframe(df_hasil.head())
        st.subheader("--- AKHIR DEBUG: PREDICT ---")
    return df_hasil

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard Lanjutan", layout="wide", page_icon="🛡️")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard - Analisis Dataset")

if DEBUG_MODE:
    st.warning("MODE DEBUG AKTIF. Output tambahan akan ditampilkan.")

model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV (pemisah ';') atau Excel", type=["csv", "xlsx", "xls"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("PENTING: Nama Kolom di File Anda")
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
        Kesalahan nama kolom akan menyebabkan hasil analisis tidak akurat.
        """
    )

    if uploaded_file is not None:
        try:
            df_input_original = None # Inisialisasi
            if uploaded_file.name.endswith('.csv'):
                df_input_original = pd.read_csv(uploaded_file, sep=';')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df_input_original = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
                st.stop()
            
            if DEBUG_MODE and df_input_original is not None:
                st.subheader("--- DEBUG: DATA INPUT ASLI ---")
                st.write("Kolom terdeteksi di file unggahan:")
                st.write(df_input_original.columns.tolist())
                st.write("Lima baris pertama dari file yang diunggah:")
                st.dataframe(df_input_original.head())
                st.subheader("--- AKHIR DEBUG: DATA INPUT ASLI ---")


            st.subheader("📄 Data Awal yang Diunggah (Contoh 5 Baris Pertama):")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis pada Seluruh Dataset", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data... ⏳"):
                    # Penting: kirim salinan df_input_original ke preprocess_dataframe
                    df_processed_for_model = preprocess_dataframe(df_input_original.copy())
                    
                    if df_processed_for_model.empty:
                        st.warning("Tidak ada data yang bisa diproses setelah pra-pemrosesan.")
                    else:
                        df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)
                        
                        if df_prediksi_hasil.empty:
                            st.warning("Tidak ada hasil prediksi yang bisa ditampilkan.")
                        else:
                            # Gabungkan dengan hati-hati, pastikan indeks selaras
                            df_tampilan_akhir = pd.concat([
                                df_input_original.reset_index(drop=True),
                                df_prediksi_hasil.reset_index(drop=True)
                            ], axis=1)

                            st.subheader("📊 Hasil Analisis Lengkap per Data:")
                            # Ambil semua kolom dari df_input_original dan kolom prediksi yang relevan
                            kolom_input_asli_untuk_tampil = list(df_input_original.columns)
                            kolom_prediksi_untuk_tampil = ['Status Deteksi', 'Probabilitas Ancaman (%)'] # Sebelumnya 'Tingkat Kepercayaan Ancaman (%)'
                            kolom_tampil_final = kolom_input_asli_untuk_tampil + kolom_prediksi_untuk_tampil
                            
                            # Pastikan kolom ada sebelum mencoba menampilkannya
                            kolom_tampil_final = [kol for kol in kolom_tampil_final if kol in df_tampilan_akhir.columns]

                            st.dataframe(df_tampilan_akhir[kolom_tampil_final], use_container_width=True)
                            st.markdown("---")
                            st.subheader("📈 Ringkasan dan Statistik Hasil Analisis:")

                            # ... (Sisa kode untuk metrik dan visualisasi sama seperti sebelumnya) ...
                            jumlah_total_data = len(df_tampilan_akhir)
                            # Gunakan 'Probabilitas Ancaman (%)' untuk konsistensi nama kolom
                            jumlah_terancam = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                            jumlah_aman = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Aman'])
                            persentase_terancam = (jumlah_terancam / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0
                            persentase_aman = (jumlah_aman / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(label="Total Data Dianalisis", value=f"{jumlah_total_data} baris")
                            with col2:
                                st.metric(label="🚨 Data Terdeteksi Ancaman", value=f"{jumlah_terancam} baris",
                                        delta=f"{persentase_terancam:.2f}% dari total", delta_color="inverse")
                            with col3:
                                st.metric(label="✅ Data Terdeteksi Aman", value=f"{jumlah_aman} baris",
                                        delta=f"{persentase_aman:.2f}% dari total", delta_color="normal")

                            if jumlah_total_data > 0:
                                st.subheader("Visualisasi Hasil:")
                                st.write("Perbandingan Status Deteksi:")
                                data_grafik_status_df = pd.DataFrame({
                                    'Status': ['Terancam', 'Aman'],
                                    'Jumlah': [jumlah_terancam, jumlah_aman]
                                })
                                st.bar_chart(data_grafik_status_df.set_index('Status'), color=["#FF4B4B", "#3DDC97"])

                                if jumlah_terancam > 0:
                                    st.write("Distribusi Probabilitas Ancaman (%) untuk Data Terancam:")
                                    # Gunakan 'Probabilitas Ancaman (%)' untuk konsistensi nama kolom
                                    data_terancam_probs = df_tampilan_akhir[
                                        df_tampilan_akhir['Status Deteksi'] == 'Terancam'
                                    ]['Probabilitas Ancaman (%)'] 
                                    
                                    g_probs = sns.displot(data_terancam_probs, kind="hist", kde=True, color="#FF4B4B")
                                    g_probs.set_axis_labels("Probabilitas Ancaman (%)", "Jumlah")
                                    g_probs.fig.suptitle('Distribusi Probabilitas Ancaman (Data Terancam)', y=1.03)
                                    st.pyplot(g_probs.fig)
                                
                                if INPUT_COL_PACKET_SIZE in df_tampilan_akhir.columns: # Periksa menggunakan nama kolom input asli
                                    st.subheader(f"Analisis '{INPUT_COL_PACKET_SIZE}':")
                                    g_packet = sns.displot(data=df_tampilan_akhir, 
                                                        x=INPUT_COL_PACKET_SIZE, 
                                                        hue='Status Deteksi', 
                                                        kind="hist", kde=True,
                                                        palette={"Aman":"#3DDC97", "Terancam":"#FF4B4B"})
                                    g_packet.set_axis_labels(f"{INPUT_COL_PACKET_SIZE}", "Jumlah")
                                    g_packet.fig.suptitle(f'Distribusi {INPUT_COL_PACKET_SIZE} berdasarkan Status Deteksi', y=1.03)
                                    st.pyplot(g_packet.fig)
        except pd.errors.ParserError:
            st.error("Gagal mem-parsing file. Pastikan format CSV benar dan pemisah kolom adalah titik koma (';').")
        except KeyError as e:
            st.error(f"Kolom yang dibutuhkan tidak ditemukan saat pemrosesan: {e}. "
                       "Ini bisa terjadi di dalam fungsi pra-pemrosesan atau saat mengakses kolom di DataFrame. "
                       "Periksa kembali nama kolom di file Anda dan pastikan sesuai dengan yang diharapkan "
                       f"(lihat bagian 'PENTING: Nama Kolom' di sidebar). Juga periksa logika `row.get()`.")
        except Exception as e:
            st.error(f"Terjadi kesalahan umum: {e}")
            st.exception(e) # Tampilkan traceback untuk debugging lebih lanjut
    else:
        st.info("Unggah dataset (CSV dengan pemisah ';' atau Excel) melalui panel kiri.")
else:
    st.warning("Model tidak berhasil dimuat.")

st.markdown("---")
st.caption("Dashboard IDS v0.6 (Debug Mode) | Dibuat dengan Streamlit & Seaborn")
