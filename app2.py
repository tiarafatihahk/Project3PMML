# ======================================================================================
# app.py - VERSI FINAL DENGAN 12 FITUR YANG BENAR
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import datetime

# ======================================================================================
# KONFIGURASI PUSAT - Sesuai dengan output notebook Anda
# ======================================================================================

MODEL_FILENAME = 'best_random_forest_model (2).pkl'

# INI ADALAH DAFTAR 12 FITUR YANG PASTI DAN BENAR DARI NOTEBOOK ANDA
FEATURE_ORDER = [
    'num__login_attempts',
    'num__failed_logins',
    'num__ip_reputation_score',
    'num__network_packet_size',
    'cat__unusual_time_access_Tidak',
    'cat__unusual_time_access_Ya',
    'cat__browser_type_Chrome',
    'cat__browser_type_Firefox',
    'cat__protocol_type_TCP',
    'cat__protocol_type_UDP',
    'cat__encryption_used_TLS',
    'cat__encryption_used_Tidak Ada',
]

# ======================================================================================
# FUNGSI UTAMA
# ======================================================================================

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        return None
    try:
        model = joblib.load(model_path)
        # Verifikasi terakhir untuk memastikan
        if model.n_features_in_ != len(FEATURE_ORDER):
            st.error(f"FATAL: Model butuh {model.n_features_in_} fitur, tapi FEATURE_ORDER hanya punya {len(FEATURE_ORDER)}.")
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

model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Input Parameter Aktivitas Jaringan:")
    
    # Kumpulkan pilihan pengguna di sini
    # Hanya minta input untuk fitur ASLI (sebelum di-encode)
    user_selections = {}
    user_selections['num__login_attempts'] = st.sidebar.number_input("Jumlah Percobaan Login", value=3)
    user_selections['num__failed_logins'] = st.sidebar.number_input("Jumlah Kegagalan Login", value=1)
    user_selections['num__ip_reputation_score'] = st.sidebar.slider("Skor Reputasi IP", 0, 100, 80)
    user_selections['num__network_packet_size'] = st.sidebar.number_input("Ukuran Paket Jaringan (KB)", value=1.5)
    
    # Opsi HARUS SAMA dengan yang ada di nama fitur Anda
    user_selections['cat__unusual_time_access'] = st.sidebar.radio("Akses di Waktu Tidak Biasa?", ("Tidak", "Ya"))
    user_selections['cat__browser_type'] = st.sidebar.selectbox("Tipe Browser", ["Chrome", "Firefox"]) # Hanya ada 2 opsi ini di model Anda
    user_selections['cat__protocol_type'] = st.sidebar.selectbox("Tipe Protokol", ["TCP", "UDP"]) # Hanya ada 2 opsi ini
    user_selections['cat__encryption_used'] = st.sidebar.selectbox("Tipe Enkripsi", ["TLS", "Tidak Ada"]) # Hanya ada 2 opsi ini

    if st.sidebar.button("Analisis Aktivitas", type="primary", use_container_width=True):
        
        # Siapkan dictionary final untuk input model
        final_model_inputs = {}
        
        # Proses input untuk model
        for feature in FEATURE_ORDER:
            # Fitur numerik
            if feature.startswith('num__'):
                final_model_inputs[feature] = user_selections[feature]
            # Fitur kategorikal
            elif feature.startswith('cat__'):
                # Pisahkan nama fitur asli dan nilainya (misal: 'cat__browser_type' dan 'Chrome')
                base_feature_name, value = feature.rsplit('_', 1) 
                # Cek apakah pilihan pengguna cocok dengan nilai fitur saat ini
                final_model_inputs[feature] = 1 if user_selections[base_feature_name] == value else 0
            else:
                final_model_inputs[feature] = 0 # Safety net

        # Buat input array
        input_array = np.array([final_model_inputs[feature] for feature in FEATURE_ORDER]).reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        is_threat = (prediction[0] == 1)
        st.subheader("Hasil Analisis:")
        if is_threat:
            st.error("🔴 Terdeteksi Potensi Ancaman!", icon="🚨")
            st.metric("Tingkat Kepercayaan Ancaman", f"{prediction_proba[0][1]*100:.2f}%")
        else:
            st.success("✅ Aktivitas Jaringan Terlihat Normal.", icon="👍")
            st.metric("Tingkat Kepercayaan Normal", f"{prediction_proba[0][0]*100:.2f}%")
else:
    st.error("Aplikasi tidak bisa berjalan. Periksa log di atas.")
