# IDS Dashboard for E-Commerce (Capstone Project)

Ini adalah prototipe dashboard Intrusion Detection System (IDS) yang dibangun menggunakan Streamlit dan model Machine Learning (Random Forest) untuk mendeteksi aktivitas jaringan yang mencurigakan. Proyek ini merupakan bagian dari Capstone Project Machine Learning Kelompok 3.

## 🚀 Fitur
- Antarmuka untuk input parameter aktivitas jaringan.
- Prediksi real-time (Normal / Potensi Ancaman) menggunakan model yang sudah dilatih.
- Dashboard log aktivitas untuk memantau analisis terbaru.
- (Opsional) Sistem notifikasi via email untuk peringatan dini.

## 🛠️ Cara Menjalankan Secara Lokal

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username/nama-repo.git](https://github.com/username/nama-repo.git)
    cd nama-repo
    ```

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh File Model:** Pastikan file model `model_ids_final.pkl` sudah ada di direktori utama. (Lihat catatan di bawah tentang file model).

5.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

6.  Buka browser Anda dan akses alamat URL yang ditampilkan di terminal.

## 📸 Screenshot
(Sangat disarankan untuk menambahkan screenshot dari aplikasi Anda yang sedang berjalan di sini)
