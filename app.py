import streamlit as st
import pandas as pd
import numpy as np
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === Load model ===
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# === Fungsi kirim email ===
def kirim_email(ip, aktivitas):
    sender = "akunemailmu@gmail.com"
    receiver = "penerima@gmail.com"
    password = "app_password_gmailmu"  # Gunakan app password Gmail

    subject = f"⚠️ Peringatan Aktivitas Mencurigakan dari IP {ip}"
    body = f"Aktivitas mencurigakan terdeteksi:\n\nIP: {ip}\nAktivitas: {aktivitas}"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        st.success("Email notifikasi berhasil dikirim.")
    except Exception as e:
        st.error(f"Gagal mengirim email: {e}")

# === Load data log ===
@st.cache_data
def load_data():
    return pd.read_csv('log_aktivitas.csv')

df = load_data()

# === Prediksi aktivitas mencurigakan ===
fitur_model = df[['fitur1', 'fitur2', 'fitur3']]  # ganti sesuai kolom fitur
prediksi = model.predict(fitur_model)
df['Status'] = np.where(prediksi == 1, 'Mencurigakan', 'Normal')

# === Streamlit UI ===
st.title("🛡️ Dashboard IDS E-Commerce")

st.subheader("📋 Log Aktivitas Terbaru")
st.dataframe(df)

# === Filter mencurigakan ===
df_mencurigakan = df[df['Status'] == 'Mencurigakan']

st.subheader("🚨 Aktivitas Mencurigakan Terdeteksi")
st.write(f"Jumlah aktivitas mencurigakan: {len(df_mencurigakan)}")
st.dataframe(df_mencurigakan)

# === Daftar IP untuk pemblokiran ===
st.subheader("⛔ Daftar IP Mencurigakan")
daftar_ip = df_mencurigakan['ip'].unique()
st.write(daftar_ip)

# === Kirim notifikasi jika ada mencurigakan ===
if st.button("🔔 Kirim Notifikasi Email untuk IP Mencurigakan"):
    if len(df_mencurigakan) > 0:
        for i, row in df_mencurigakan.iterrows():
            kirim_email(row['ip'], row['aktivitas'])
    else:
        st.info("Tidak ada aktivitas mencurigakan saat ini.")

