import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
import datetime

# ======================================================================================
# KONFIGURASI APLIKASI DAN MODEL
# ======================================================================================

st.set_page_config(
    page_title="IDS Dashboard for E-Commerce",
    page_icon="🛡️",
    layout="wide"
)

# Judul dan deskripsi
st.title("🛡️ Intrusion Detection System (IDS) Dashboard")
st.markdown("""
Selamat datang di Dashboard IDS untuk E-Commerce UKM. Aplikasi ini membantu mendeteksi aktivitas jaringan yang mencurigakan atau berbahaya secara real-time untuk melindungi bisnis Anda.
Masukkan parameter aktivitas jaringan di sidebar kiri dan klik tombol "Analisis Aktivitas".
""")

# --- NAMA FILE MODEL -------
MODEL_FILENAME = 'best_random_forest_model.pkl'

# --- DAFTAR FITUR YANG BENAR (DENGAN KOMA YANG SUDAH DIPERBAIKI) ---
FEATURE_ORDER = [
    'network_packet_size',
    'login_attempts',
    'ip_reputation_score',
    'failed_logins',
    'unusual_time_access',
    'browser_type_Chrome',
    'browser_type_Edge',
    'browser_type_Firefox',
    'browser_type_Safari',
    'browser_type_Unknown', # Koma ditambahkan di sini
    'protocol_type_ICMP',
    'protocol_type_TCP',
    'protocol_type_UDP',
    'encryption_used_AES',
    'encryption_used_DES',
    'encryption_used_unencrypted'
]
# --------------------------------------------------------------------

# ======================================================================================
# FUNGSI-FUNGSI UTAMA
# ======================================================================================

@st.cache_resource
def load_model(model_path):
    """Memuat model machine learning dari file .pkl."""
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi send_alert_email (tidak diubah)
def send_alert_email(prediction_result, probability, user_inputs):
    # ... (kode email tetap sama) ...
    pass

# Inisialisasi session state untuk menyimpan log
if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# ======================================================================================
# ANTARMUKA PENGGUNA (UI) - SIDEBAR INPUT
# ======================================================================================
model = load_model(MODEL_FILENAME) # <-- Kesalahan pemanggilan diperbaiki

if model:
    st.sidebar.header("Input Parameter Aktivitas Jaringan:")
    user_inputs = {}

    # Input Fields
    user_inputs['network_packet_size'] = st.sidebar.number_input("Ukuran Paket Jaringan Rata-rata (KB)", min_value=0.0, value=1.5, step=0.1, format="%.2f")
    user_inputs['login_attempts'] = st.sidebar.number_input("Jumlah Percobaan Login (dalam 5 menit terakhir)", min_value=0, value=3, step=1)
    user_inputs['ip_reputation_score'] = st.sidebar.slider("Skor Reputasi IP (0=Buruk, 100=Baik)", 0, 100, 80)
    user_inputs['failed_logins'] = st.sidebar.number_input("Jumlah Kegagalan Login (dalam 5 menit terakhir)", min_value=0, value=1, step=1)
    
    # Categorical Features
    unusual_time_map = {"Tidak": 0, "Ya": 1}
    user_inputs['unusual_time_access'] = unusual_time_map[st.sidebar.radio("Akses di Waktu Tidak Biasa?", ("Tidak", "Ya"))]

    browser_options = ["Chrome", "Firefox", "Safari", "Edge", "Unknown"]
    selected_browser = st.sidebar.selectbox("Tipe Browser", options=browser_options)

    protocol_options = ["TCP", "UDP", "ICMP"] # Disederhanakan sesuai FEATURE_ORDER
    selected_protocol = st.sidebar.selectbox("Tipe Protokol", options=protocol_options)

    encryption_options = ["AES", "DES", "unencrypted"] # Disederhanakan sesuai FEATURE_ORDER
    selected_encryption = st.sidebar.selectbox("Jenis Enkripsi yang Digunakan", options=encryption_options)

    # Tombol Analisis
    if st.sidebar.button("Analisis Aktivitas", type="primary", use_container_width=True):
        final_model_inputs = {}

        # Daftar fitur numerik asli (sebelum one-hot encoding)
        numeric_features_original = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins']

        for feature in FEATURE_ORDER:
            # Salin nilai untuk fitur numerik
            if feature in numeric_features_original:
                final_model_inputs[feature] = user_inputs.get(feature, 0)
            
            # Buat nilai 0/1 untuk fitur 'unusual_time_access'
            elif feature == 'unusual_time_access':
                 final_model_inputs[feature] = user_inputs.get(feature, 0)

            # Buat nilai 0/1 untuk 'browser_type'
            elif 'browser_type_' in feature:
                browser_name = feature.split('browser_type_')[1]
                final_model_inputs[feature] = 1 if selected_browser == browser_name else 0

            # Buat nilai 0/1 untuk 'protocol_type'
            elif 'protocol_type_' in feature:
                protocol_name = feature.split('protocol_type_')[1]
                final_model_inputs[feature] = 1 if selected_protocol == protocol_name else 0
            
            # Buat nilai 0/1 untuk 'encryption_used'
            elif 'encryption_used_' in feature:
                encryption_name = feature.split('encryption_used_')[1]
                final_model_inputs[feature] = 1 if selected_encryption == encryption_name else 0

            else:
                final_model_inputs[feature] = 0

        # Buat input array
        input_array = np.array([final_model_inputs[feature] for feature in FEATURE_ORDER]).reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        # Simpan log
        is_threat = (prediction[0] == 1)
        threat_probability = prediction_proba[0][1] if is_threat else 1 - prediction_proba[0][0]
        
        log_entry = {
            "Waktu": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Status": "Potensi Ancaman" if is_threat else "Normal",
            "Probabilitas": f"{threat_probability*100:.2f}%",
            "Detail": f"Login Gagal: {user_inputs['failed_logins']}, Reputasi IP: {user_inputs['ip_reputation_score']}"
        }
        st.session_state.activity_log.insert(0, log_entry)
        
        # Tampilkan hasil di halaman utama
        col1, col2 = st.columns([1, 4])
        with col1:
             if is_threat:
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Icon_Danger.svg/2048px-Icon_Danger.svg.png", width=100)
             else:
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Check_mark_icon.svg/2048px-Check_mark_icon.svg.png", width=100)
        with col2:
            if is_threat:
                st.error(f"**TERDETEKSI POTENSI ANCAMAN!**", icon="🚨")
                st.metric(label="Tingkat Kepercayaan Ancaman", value=f"{prediction_proba[0][1]*100:.2f}%")
                st.warning("**Rekomendasi Tindakan:** Periksa log firewall, analisis lalu lintas dari IP terkait, dan pertimbangkan untuk memblokir sementara akses dari sumber yang mencurigakan.")
            else:
                st.success("**Aktivitas Jaringan Terlihat Normal**", icon="✅")
                st.metric(label="Tingkat Kepercayaan Normal", value=f"{prediction_proba[0][0]*100:.2f}%")
                st.info("Tidak ada tindakan segera yang diperlukan. Sistem akan terus memantau aktivitas.")

# ======================================================================================
# TAMPILAN UTAMA - LOG AKTIVITAS
# ======================================================================================

st.markdown("---")
st.subheader("📜 Log Aktivitas Real-time")
st.markdown("Menampilkan hasil analisis terbaru di bagian atas.")

if not st.session_state.activity_log:
    st.info("Belum ada aktivitas yang dianalisis. Silakan masukkan data di sidebar dan klik 'Analisis Aktivitas'.")
else:
    log_df = pd.DataFrame(st.session_state.activity_log)
    def style_status(val):
        color = 'red' if val == "Potensi Ancaman" else 'green'
        return f'color: {color}; font-weight: bold;'
    st.dataframe(log_df.style.applymap(style_status, subset=['Status']), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini merupakan bagian dari Capstone Project Machine Learning Kelompok 3.")
