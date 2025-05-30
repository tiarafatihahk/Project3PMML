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

# Judul dan deskripsi dari presentasi Anda
st.title("🛡️ Intrusion Detection System (IDS) Dashboard")
st.markdown("""
Selamat datang di Dashboard IDS untuk E-Commerce UKM. Aplikasi ini membantu mendeteksi aktivitas jaringan yang mencurigakan atau berbahaya secara real-time untuk melindungi bisnis Anda[cite: 4, 7].
Masukkan parameter aktivitas jaringan di sidebar kiri dan klik tombol "Analisis Aktivitas".
""")

# --- SESUAIKAN BAGIAN INI DENGAN PROYEK ANDA ---
MODEL_FILENAME = 'best_random_forest_model.pkl'

# Ganti daftar ini dengan fitur dan urutan yang sama persis seperti saat training model IDS Anda.
# Saya menggunakan contoh fitur yang umum untuk kasus ini.
FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score',
    'failed_logins', 'unusual_time_access', 'browser_type_Chrome', 
    'browser_type_Edge', 'browser_type_Firefox', 'browser_type_Safari',
    'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
# ----------------------------------------------------

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

def send_alert_email(prediction_result, probability, user_inputs):
    """Mengirim notifikasi email jika terdeteksi ancaman."""
    # Gunakan st.secrets untuk menyimpan kredensial email dengan aman
    try:
        sender_email = st.secrets["email"]["sender_email"]
        receiver_email = st.secrets["email"]["receiver_email"]
        password = st.secrets["email"]["password"]
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = st.secrets["email"]["smtp_port"]
    except KeyError:
        st.warning("Konfigurasi email tidak ditemukan di st.secrets. Notifikasi email dinonaktifkan.")
        return

    subject = f"🚨 PERINGATAN KEAMANAN: Terdeteksi Potensi Ancaman!"
    body = f"""
    Halo Tim Keamanan,

    Sistem IDS kami telah mendeteksi aktivitas jaringan yang mencurigakan.

    - Hasil Prediksi: {prediction_result}
    - Tingkat Kepercayaan (Probabilitas Ancaman): {probability*100:.2f}%
    - Waktu Deteksi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Detail Aktivitas:
    """
    for key, value in user_inputs.items():
        body += f"- {key.replace('_', ' ').title()}: {value}\n"

    body += "\nMohon segera lakukan investigasi lebih lanjut.\n\nTerima kasih,\nSistem Monitoring IDS Otomatis"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        st.toast("Notifikasi peringatan berhasil dikirim via email!", icon="📧")
    except Exception as e:
        st.error(f"Gagal mengirim email: {e}")

# Inisialisasi session state untuk menyimpan log
if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# ======================================================================================
# ANTARMUKA PENGGUNA (UI) - SIDEBAR INPUT
# ======================================================================================
model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Input Parameter Aktivitas Jaringan:")
    user_inputs = {}

    # Input Fields (sesuaikan dengan fitur Anda)
    user_inputs['login_attempts'] = st.sidebar.number_input("Jumlah Percobaan Login (dalam 5 menit terakhir)", min_value=0, value=3, step=1)
    user_inputs['failed_logins'] = st.sidebar.number_input("Jumlah Kegagalan Login (dalam 5 menit terakhir)", min_value=0, value=1, step=1)
    user_inputs['ip_reputation_score'] = st.sidebar.slider("Skor Reputasi IP (0=Buruk, 100=Baik)", 0, 100, 80)
    user_inputs['network_packet_size'] = st.sidebar.number_input("Ukuran Paket Jaringan Rata-rata (KB)", min_value=0.0, value=1.5, step=0.1, format="%.2f")

    # Categorical Features (perlu di-encode)
    unusual_time_map = {"Tidak": 0, "Ya": 1}
    user_inputs['unusual_time_access'] = unusual_time_map[st.sidebar.radio("Akses di Waktu Tidak Biasa?", ("Tidak", "Ya"))]

    browser_options = ["Chrome", "Firefox", "Safari", "Edge", "Lainnya/Unknown"]
    browser_map = {name: i for i, name in enumerate(browser_options)}
    selected_browser = st.sidebar.selectbox("Tipe Browser", options=browser_options)
    user_inputs['browser_type'] = browser_map.get(selected_browser, len(browser_options)-1)

    protocol_options = ["TCP", "UDP", "HTTP", "HTTPS", "ICMP"]
    protocol_map = {name: i for i, name in enumerate(protocol_options)}
    selected_protocol = st.sidebar.selectbox("Tipe Protokol", options=protocol_options)
    user_inputs['protocol_type'] = protocol_map.get(selected_protocol, 0)

    encryption_options = ["AES", "DES", "TLS", "Tidak Ada"]
    encryption_map = {name: i for i, name in enumerate(encryption_options)}
    selected_encryption = st.sidebar.selectbox("Jenis Enkripsi yang Digunakan", options=encryption_options)
    user_inputs['encryption_used'] = encryption_map.get(selected_encryption, len(encryption_options)-1)

    # Tombol Analisis
    if st.sidebar.button("Analisis Aktivitas", type="primary", use_container_width=True):
        # Susun input sesuai urutan fitur model
        input_array = np.array([user_inputs[feature] for feature in FEATURE_ORDER]).reshape(1, -1)

        # Prediksi
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        # Simpan hasil untuk ditampilkan
        is_threat = (prediction[0] == 1) # Asumsi 1 adalah ancaman/malicious
        threat_probability = prediction_proba[0][1] if is_threat else prediction_proba[0][0]

        # Simpan log ke session state
        log_entry = {
            "Waktu": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Status": "Potensi Ancaman" if is_threat else "Normal",
            "Probabilitas": f"{threat_probability*100:.2f}%",
            "Detail": f"Login Gagal: {user_inputs['failed_logins']}, Reputasi IP: {user_inputs['ip_reputation_score']}"
        }
        st.session_state.activity_log.insert(0, log_entry) # Tambahkan ke awal list

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
                # Kirim email jika terdeteksi ancaman (sesuai spesifikasi proyek [cite: 16])
                # send_alert_email("Potensi Ancaman Terdeteksi", prediction_proba[0][1], user_inputs)
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
    # Tambahkan styling berdasarkan status
    def style_status(val):
        color = 'red' if val == "Potensi Ancaman" else 'green'
        return f'color: {color}; font-weight: bold;'
    st.dataframe(log_df.style.applymap(style_status, subset=['Status']), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini merupakan bagian dari Capstone Project Machine Learning Kelompok 3.")
