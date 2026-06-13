import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. Pengaturan Konfigurasi Halaman Web
st.set_page_config(page_title="Spotify Music Recommender", layout="wide", page_icon="🎵")

st.title("🎵 Sistem Rekomendasi Musik Spotify")
st.write("Tugas Besar Machine Learning - Kelompok CRISP-DM")

# 2. Memuat Data yang Sudah Diproses di Google Colab
@st.cache_data
def load_data():
    # Membaca berkas hasil ekspor Tahap 6 Deployment di Colab
    data = pd.read_csv('spotify_processed_data.csv')
    return data

try:
    df = load_data()

    # 3. Menyiapkan Fitur Audio untuk Content-Based Filtering
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[audio_features])
    
    # Menghitung matriks kesamaan Cosine Similarity
    cosine_sim = cosine_similarity(scaled_features, scaled_features)

    # 4. UI Dropdown Interaktif untuk Memilih Lagu
    st.subheader("Pilih Lagu Favorit Kamu")
    list_lagu = df['track_name'].values
    lagu_dipilih = st.selectbox("Cari atau pilih judul lagu di bawah ini:", list_lagu)

    # 5. Tombol Proses Rekomendasi
    if st.button("Dapatkan Rekomendasi Musik"):
        # Mencari indeks dari lagu yang dipilih
        idx = df[df['track_name'] == lagu_dipilih].index[0]
        
        # Menghitung skor kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Mengambil 5 lagu paling mirip (indeks 0 adalah lagu itu sendiri)
        top_5_songs = sim_scores[1:6]
        
        st.success(f"Berikut adalah 5 lagu rekomendasi yang mirip dengan '{lagu_dipilih}':")
        
        # Menampilkan hasil dalam bentuk kolom yang rapi
        for i, (index_lagu, skor) in enumerate(top_5_songs, 1):
            nama_lagu = df['track_name'].iloc[index_lagu]
            artis = df['artists'].iloc[index_lagu]
            album = df['album_name'].iloc[index_lagu]
            st.write(f"**{i}. {nama_lagu}** oleh *{artis}* (Album: {album}) — Skor Kemiripan: {round(skor*100, 2)}%")

except FileNotFoundError:
    st.error("Berkas 'spotify_processed_data.csv' tidak ditemukan! Pastikan kamu sudah menjalankan Google Colab sampai Tahap 6 dan mengunduh berkas hasilnya ke dalam folder yang sama dengan file app.py ini.")