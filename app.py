import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ======================================================================================
# KONFIGURASI PUSAT
# ======================================================================================
MODEL_FILENAME = 'model_final.pkl'
SCALER_FILENAME = 'scaler_final.pkl'

FEATURE_ORDER = [
    'network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins',
    'unusual_time_access', 'browser_type_Chrome', 'browser_type_Edge', 'browser_type_Firefox',
    'browser_type_Safari', 'browser_type_Unknown', 'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'encryption_used_AES', 'encryption_used_DES', 'encryption_used_unencrypted'
]
NUMERIC_FEATURES = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins']

INPUT_COL_LOGIN_ATTEMPTS = 'login_attempts'
INPUT_COL_FAILED_LOGINS = 'failed_logins'
INPUT_COL_IP_REPUTATION = 'ip_reputation_score'
INPUT_COL_PACKET_SIZE = 'network_packet_size'
INPUT_COL_UNUSUAL_TIME = 'unusual_time_access'
INPUT_COL_BROWSER = 'browser_type'
INPUT_COL_PROTOCOL = 'protocol_type'
INPUT_COL_ENCRYPTION = 'encryption_used'

BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS_DROPDOWN = ["AES", "DES", "None"]
ENCRYPTION_OPTIONS_MODEL = ["AES", "DES", "unencrypted"]

# ======================================================================================
# FUNGSI-FUNGSI BANTU
# ======================================================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        st.error(f"File scaler '{scaler_path}' tidak ditemukan!")
        return None
    return joblib.load(scaler_path)

def preprocess_dataframe(df_input, scaler):
    df = df_input.copy()
    required_cols = NUMERIC_FEATURES + [INPUT_COL_UNUSUAL_TIME, INPUT_COL_BROWSER, INPUT_COL_PROTOCOL, INPUT_COL_ENCRYPTION]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col in NUMERIC_FEATURES else "Unknown"

    numeric_data = df[NUMERIC_FEATURES]
    scaled_numeric_data = scaler.transform(numeric_data)
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=NUMERIC_FEATURES, index=df.index)

    categorical_features = {}
    categorical_features['unusual_time_access'] = df[INPUT_COL_UNUSUAL_TIME].apply(lambda x: 1 if x == "Ya" else 0)
    for option in BROWSER_OPTIONS:
        categorical_features[f'browser_type_{option}'] = df[INPUT_COL_BROWSER].apply(lambda x: 1 if x == option else 0)
    for option in PROTOCOL_OPTIONS:
        categorical_features[f'protocol_type_{option}'] = df[INPUT_COL_PROTOCOL].apply(lambda x: 1 if x == option else 0)
    encryption_mapped = df[INPUT_COL_ENCRYPTION].apply(lambda x: "unencrypted" if x == "None" else x)
    for option in ENCRYPTION_OPTIONS_MODEL:
        categorical_features[f"encryption_used_{option}"] = encryption_mapped.apply(lambda x: 1 if x == option else 0)
    df_categorical_processed = pd.DataFrame(categorical_features, index=df.index)

    df_combined = pd.concat([df_scaled, df_categorical_processed], axis=1)
    df_processed = df_combined.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df_processed

def model_prediksi_ancaman_dataset(model, df_processed):
    if df_processed.empty: return pd.DataFrame()
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)
    df_hasil = pd.DataFrame({
        'Status Deteksi': ['Terancam' if p == 1 else 'Aman' for p in predictions],
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    return df_hasil

# --- FUNGSI BARU UNTUK MENGIRIM EMAIL ---
def send_email_alert(recipient, subject, body, config):
    if not config['sender_email'] or not config['sender_password']:
        st.sidebar.warning("Harap isi email dan password pengirim.")
        return

    msg = MIMEMultipart()
    msg['From'] = config['sender_email']
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])
        text = msg.as_string()
        server.sendmail(config['sender_email'], recipient, text)
        server.quit()
        st.sidebar.success("Notifikasi email berhasil dikirim!")
    except Exception as e:
        st.sidebar.error(f"Gagal mengirim email: {e}")

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Dashboard Intrusion Detection System (IDS)")

model = load_model(MODEL_FILENAME)
scaler = load_scaler(SCALER_FILENAME)

# --- Sidebar ---
st.sidebar.header("⚙️ Opsi Analisis")
analysis_mode = st.sidebar.radio("Pilih Mode Analisis:", ('Unggah File', 'Input Manual'))
st.sidebar.markdown("---")

# --- Bagian Baru untuk Notifikasi Email ---
with st.sidebar.expander("🔔 Pengaturan Notifikasi Email"):
    enable_email = st.checkbox("Aktifkan Notifikasi Email")
    recipient_email = st.text_input("Email Penerima Peringatan")
    sender_email = st.text_input("Email Pengirim (misal: akun Gmail Anda)")
    sender_password = st.text_input("Password Aplikasi Pengirim", type="password")
    st.warning(
        "**Penting:** Untuk akun Gmail, gunakan 'App Password' yang dibuat dari pengaturan keamanan akun Google Anda, bukan password login biasa.",
        icon="⚠️"
    )
email_config = {"sender_email": sender_email, "sender_password": sender_password}


# --- Tampilan Utama ---
if not model or not scaler:
    st.error("Gagal memuat `model_terbaik.pkl` dan/atau `scaler.pkl`.")
else:
    if analysis_mode == 'Input Manual':
        st.sidebar.subheader("Form Prediksi Manual")
        with st.sidebar.form(key='manual_input_form'):
            login_attempts = st.number_input('Percobaan Login', min_value=0, value=3)
            failed_logins = st.number_input('Login Gagal', min_value=0, value=1)
            ip_reputation_score = st.slider('Skor Reputasi IP', 0, 100, 50)
            network_packet_size = st.number_input('Ukuran Paket Jaringan', min_value=0, value=512)
            unusual_time_access = st.selectbox('Akses Waktu Tidak Wajar?', ('Tidak', 'Ya'))
            browser_type = st.selectbox('Browser', BROWSER_OPTIONS)
            protocol_type = st.selectbox('Protokol', PROTOCOL_OPTIONS)
            encryption_used = st.selectbox('Enkripsi', ENCRYPTION_OPTIONS_DROPDOWN)
            submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

        st.info("Isi formulir di sidebar kiri dan klik 'Prediksi Sekarang' untuk melihat hasilnya di sini.")
        if submit_button:
            manual_data = {
                INPUT_COL_LOGIN_ATTEMPTS: [login_attempts], INPUT_COL_FAILED_LOGINS: [failed_logins],
                INPUT_COL_IP_REPUTATION: [ip_reputation_score / 100.0], INPUT_COL_PACKET_SIZE: [network_packet_size],
                INPUT_COL_UNUSUAL_TIME: [unusual_time_access], INPUT_COL_BROWSER: [browser_type],
                INPUT_COL_PROTOCOL: [protocol_type], INPUT_COL_ENCRYPTION: [encryption_used]
            }
            df_manual = pd.DataFrame.from_dict(manual_data)

            with st.spinner("Melakukan prediksi..."):
                df_processed = preprocess_dataframe(df_manual, scaler)
                df_prediction = model_prediksi_ancaman_dataset(model, df_processed)
                status = df_prediction['Status Deteksi'].iloc[0]
                probabilitas = df_prediction['Probabilitas Ancaman (%)'].iloc[0]
                st.markdown("---"); st.subheader("🔍 Hasil Prediksi Manual")
                if status == 'Terancam':
                    st.error(f"**Status: Terdeteksi Ancaman**"); st.metric(label="Tingkat Kepercayaan Ancaman", value=f"{probabilitas}%")
                    if enable_email and recipient_email:
                        email_subject = "Peringatan Keamanan: Ancaman Terdeteksi (Manual Input)"
                        email_body = f"""
                        Peringatan, sistem IDS mendeteksi aktivitas mencurigakan dengan detail sebagai berikut:
                        - Status: {status}
                        - Probabilitas Ancaman: {probabilitas}%
                        - Percobaan Login: {login_attempts}
                        - Login Gagal: {failed_logins}
                        - Skor Reputasi IP: {ip_reputation_score}
                        - Waktu Akses Tidak Wajar: {unusual_time_access}
                        - Browser: {browser_type}
                        
                        Harap segera lakukan investigasi.
                        """
                        send_email_alert(recipient_email, email_subject, email_body, email_config)
                else:
                    st.success(f"**Status: Aman**"); st.metric(label="Tingkat Kepercayaan Ancaman", value=f"{probabilitas}%")

    elif analysis_mode == 'Unggah File':
        uploaded_file = st.file_uploader("Unggah Dataset (CSV atau Excel)", type=["csv", "xlsx"])
        if uploaded_file is None:
            st.info("Silakan unggah file dataset untuk memulai analisis batch.")
        else:
            try:
                df_input_original = pd.read_csv(uploaded_file, sep=';') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.subheader("📄 Pratinjau Data yang Diunggah")
                st.dataframe(df_input_original.head())
                if st.button("🚀 Jalankan Analisis File", type="primary"):
                    with st.spinner("Menganalisis data..."):
                        df_processed = preprocess_dataframe(df_input_original, scaler)
                        df_prediction = model_prediksi_ancaman_dataset(model, df_processed)
                        df_final = pd.concat([df_input_original.reset_index(drop=True), df_prediction.reset_index(drop=True)], axis=1)
                        
                        st.markdown("---"); st.subheader("📊 Dashboard Hasil Analisis")
                        total_data = len(df_final); jumlah_ancaman = len(df_final[df_final['Status Deteksi'] == 'Terancam']); jumlah_aman = total_data - jumlah_ancaman
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Data", f"{total_data}"); col2.metric("🚨 Ancaman", f"{jumlah_ancaman}"); col3.metric("✅ Aman", f"{jumlah_aman}")
                        st.markdown("<br>", unsafe_allow_html=True)

                        col_viz1, col_viz2 = st.columns(2)
                        with col_viz1:
                            st.write("**Proporsi Status Deteksi**")
                            if total_data > 0:
                                status_counts = df_final['Status Deteksi'].value_counts()
                                fig1, ax1 = plt.subplots(figsize=(5, 4)); ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FF4B4B', '#3DDC97']); ax1.axis('equal'); st.pyplot(fig1)
                        with col_viz2:
                            st.write("**Ancaman per Browser**")
                            if jumlah_ancaman > 0:
                                st.bar_chart(df_final[df_final['Status Deteksi'] == 'Terancam'][INPUT_COL_BROWSER].value_counts())
                        
                        st.write("**Distribusi Skor Reputasi IP**"); fig2, ax2 = plt.subplots(); sns.kdeplot(data=df_final, x=INPUT_COL_IP_REPUTATION, hue='Status Deteksi', fill=True, palette={'Aman': '#3DDC97', 'Terancam': '#FF4B4B'}, ax=ax2); st.pyplot(fig2)

                        if enable_email and recipient_email and jumlah_ancaman > 0:
                            email_subject = f"Peringatan Keamanan: {jumlah_ancaman} Ancaman Terdeteksi dari Analisis File"
                            email_body = f"""
                            Laporan analisis batch dari file '{uploaded_file.name}' telah selesai.
                            Sistem mendeteksi total {jumlah_ancaman} aktivitas mencurigakan dari {total_data} data yang dianalisis.
                            Harap login ke dashboard untuk melihat detail lengkap.
                            """
                            send_email_alert(recipient_email, email_subject, email_body, email_config)
                        
                        st.markdown("---"); st.subheader("📋 Tabel Hasil Analisis Lengkap"); st.dataframe(df_final)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
