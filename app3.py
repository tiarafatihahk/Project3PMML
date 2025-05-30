# ======================================================================================
# ANTARMUKA PENGGUNA (UI) - SIDEBAR INPUT
# ======================================================================================
model = load_model(MODEL_FILENAME)

# --- BLOK DEBUGGING BARU DIMULAI DI SINI ---
if model:
    # Cek dan tampilkan jumlah fitur yang diharapkan model vs. yang disediakan
    expected_features = model.n_features_in_
    provided_features = len(FEATURE_ORDER)

    st.info(f"✅ Model berhasil dimuat.")
    st.info(f"Model ini dilatih dan mengharapkan persis **{expected_features}** fitur.")
    st.info(f"Daftar `FEATURE_ORDER` Anda saat ini menyediakan **{provided_features}** fitur.")

    # Jika jumlahnya tidak cocok, tampilkan pesan error besar dan jelas
    if expected_features != provided_features:
        st.error(
            f"**STOP! JUMLAH FITUR TIDAK COCOK!**\n\n"
            f"- Model Anda memerlukan: **{expected_features} fitur**.\n"
            f"- `FEATURE_ORDER` Anda hanya memiliki: **{provided_features} fitur**.\n\n"
            f"**Solusi:** Buka kembali file `app.py` dan perbaiki daftar `FEATURE_ORDER` agar berisi persis **{expected_features}** nama fitur dengan urutan yang benar."
        )
    else:
        # Jika sudah cocok, lanjutkan membuat form input
        st.success("✅ Jumlah fitur sudah sesuai! Silakan isi form di sidebar.")
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

        # Tombol Analisis (dan sisa kode setelahnya tetap sama)
        # ...
