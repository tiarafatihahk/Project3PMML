# ======================================================================================
# app2.py - VERSI FINAL YANG SUDAH DIPERBAIKI
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import datetime

# ======================================================================================
# KONFIGURASI PUSAT (Hanya bagian ini yang mungkin perlu Anda sesuaikan di masa depan)
# ======================================================================================

MODEL_FILENAME = 'best_random_forest_model.pkl'

# Definisikan fitur-fitur ASLI Anda SEBELUM di-one-hot-encode
NUMERIC_FEATURES = [
    'network_packet_size',
    'login_attempts',
    'ip_reputation_score',
    'failed_logins'
]

# Definisikan fitur kategorikal dan semua kemungkinan nilainya
CATEGORICAL_FEATURES_MAP = {
    'unusual_time_access': ['Tidak', 'Ya'], # Ini akan menjadi 1 fitur biner
    'browser_type': ['Chrome', 'Edge', 'Firefox', 'Safari', 'Unknown'],
    'protocol_type': ['ICMP', 'TCP', 'UDP'],
    'encryption_used': ['AES', 'DES', 'unencrypted']
}

# Fungsi untuk membuat daftar fitur yang sudah di-one-hot-encode secara otomatis
def generate_feature_order():
    feature_order = NUMERIC_FEATURES.copy()
    # 'unusual_time_access' direpresentasikan sebagai satu fitur biner (0 atau 1)
    feature_order.append('unusual_time_access') 
    
    # Buat nama fitur one-hot-encoded
    for cat, values in CATEGORICAL_FEATURES_MAP.items():
        if cat != 'unusual_time_access': # 'unusual_time_access' sudah ditangani
            for value in values:
                feature_order.append(f"{cat}_{value}")
    
    return feature_order

# Hasilkan FEATURE_ORDER secara dinamis
FEATURE_ORDER = generate_feature_order()

# ======================================================================================
# FUNGSI-FUNGSI UTAMA
# ======================================================================================

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return None
    try:
        model = joblib.load(model_path)
        # Verifikasi jumlah fitur saat model dimuat
        if model.n_features_in_ != len(FEATURE_ORDER):
            st.error(
                f"ERROR KONFIGURASI: Model mengharapkan {model.n_features_in_} fitur, "
                f"tetapi konfigurasi aplikasi menghasilkan {len(FEATURE_ORDER)} fitur. "
                f"Periksa daftar NUMERIC_FEATURES dan CATEGORICAL_FEATURES_MAP di kode."
            )
            return None
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================

st.set_page_config(page_title="IDS Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard")
st.markdown("Aplikasi ini menggunakan model Machine Learning untuk mendeteksi aktivitas jaringan yang mencurigakan.")

model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Input Parameter Aktivitas Jaringan:")
    
    # Buat dictionary untuk menampung pilihan pengguna
    user_selections = {}

    # Buat input untuk fitur numerik
    for feature in NUMERIC_FEATURES:
        user_selections[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, step=1.0)

    # Buat input untuk fitur kategorikal
    for feature, options in CATEGORICAL_FEATURES_MAP.items():
        if feature == 'unusual_time_access':
            user_selections[feature] = st.sidebar.radio("Akses di Waktu Tidak Biasa?", options, index=0)
        else:
            user_selections[feature] = st.sidebar.selectbox(f"{feature.replace('_', ' ').title()}", options=options)

    # Tombol Analisis
    if st.sidebar.button("Analisis Aktivitas", type="primary", use_container_width=True):
        
        # Lakukan One-Hot Encoding manual berdasarkan pilihan pengguna
        final_model_inputs = {}
        for feature in FEATURE_ORDER:
            if feature in NUMERIC_FEATURES:
                final_model_inputs[feature] = user_selections[feature]
            elif feature == 'unusual_time_access':
                # 'unusual_time_access' menjadi 1 fitur biner (0 atau 1)
                final_model_inputs[feature] = 1 if user_selections['unusual_time_access'] == 'Ya' else 0
            else:
                # Untuk fitur one-hot lainnya
                # Cek apakah fitur ini adalah bagian dari fitur kategorikal yang ada
                base_cat, value = feature.rsplit('_', 1)
                if base_cat in user_selections:
                    final_model_inputs[feature] = 1 if user_selections[base_cat] == value else 0
                else:
                    final_model_inputs[feature] = 0 # Safety net

        # Buat input array
        input_array = np.array([final_model_inputs[feature] for feature in FEATURE_ORDER]).reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        # Tampilkan hasil
        is_threat = (prediction[0] == 1)
        st.subheader("Hasil Analisis:")
        if is_threat:
            st.error("🔴 Terdeteksi Potensi Ancaman!", icon="🚨")
            st.metric("Tingkat Kepercayaan Ancaman", f"{prediction_proba[0][1]*100:.2f}%")
        else:
            st.success("✅ Aktivitas Jaringan Terlihat Normal.", icon="👍")
            st.metric("Tingkat Kepercayaan Normal", f"{prediction_proba[0][0]*100:.2f}%")

        # Log aktivitas (opsional)
        log_entry = {"Waktu": datetime.datetime.now().strftime('%H:%M:%S'), "Status": "Ancaman" if is_threat else "Normal", "Confidence": f"{max(prediction_proba[0])*100:.2f}%"}
        st.session_state.activity_log.insert(0, log_entry)
        st.write("---")
        st.write("Log Aktivitas Terbaru:")
        st.dataframe(pd.DataFrame(st.session_state.activity_log), use_container_width=True)

else:
    st.warning("Pastikan file model ada dan konfigurasi fitur sudah benar untuk memulai.")
