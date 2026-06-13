# Data Understanding

Pada tahap ini dilakukan eksplorasi terhadap dataset Spotify Tracks Dataset untuk memahami karakteristik data yang digunakan dalam pembangunan sistem rekomendasi musik.

Dataset berisi informasi mengenai:

- Identitas lagu (`track_id`)
- Judul lagu (`track_name`)
- Nama artis (`artists`)
- Nama album (`album_name`)
- Popularitas lagu (`popularity`)
- Fitur audio seperti:
  - Danceability
  - Energy
  - Acousticness
  - Instrumentalness
  - Liveness
  - Valence
  - Tempo

### Tampilan Dataset

Berikut merupakan hasil eksplorasi awal dataset yang digunakan.

<p align="center">
  <img src="data_understanding.png" width="900">
</p>

### 10 Artis dengan Jumlah Lagu Terbanyak

Visualisasi berikut menunjukkan artis dengan jumlah lagu terbanyak pada dataset Spotify.

<p align="center">
  <img src="10_artis_teratas.png" width="700">
</p>

### Distribusi Tingkat Energy Lagu

Distribusi fitur **energy** digunakan untuk melihat karakteristik intensitas musik pada seluruh lagu dalam dataset.

<p align="center">
  <img src="distribusi_energi.png" width="700">
</p>

---

# Data Preparation

Tahap data preparation dilakukan untuk memastikan kualitas dataset sebelum proses pemodelan.

Langkah-langkah yang dilakukan meliputi:

1. Menghapus data duplikat berdasarkan `track_id`.
2. Menghapus missing values pada atribut penting.
3. Melakukan sampling data agar penggunaan memori lebih efisien.
4. Membuat user ID simulasi.
5. Mengubah popularitas menjadi rating pengguna.
6. Melakukan feature scaling menggunakan MinMaxScaler.
7. Mengubah `track_id` menjadi indeks numerik untuk Embedding Layer.

### Hasil Data Preparation

<p align="center">
  <img src="data_preparation.png" width="900">
</p>

---

# Modeling dan Evaluation

Proyek ini menggunakan tiga pendekatan utama:

1. Content-Based Filtering
2. Collaborative Filtering (SVD)
3. Neural Network-Based Recommender System

## Arsitektur Deep Learning

Model Deep Learning dibangun menggunakan TensorFlow dan Keras Functional API dengan komponen:

- User Embedding Layer
- Track Embedding Layer
- Flatten Layer
- Concatenate Layer
- Dense Layer (128 neuron)
- Dense Layer (64 neuron)
- Dropout Layer (0.2)
- Output Layer

<p align="center">
  <img src="model_dan_evaluation_deep_learning_1.png" width="900">
</p>

## Hasil Evaluasi Model

Evaluasi dilakukan menggunakan metrik Root Mean Squared Error (RMSE). Nilai RMSE digunakan untuk mengukur selisih antara rating sebenarnya dengan rating hasil prediksi model.

<p align="center">
  <img src="model_dan_evaluation.png" width="900">
</p>

## Grafik RMSE

Grafik berikut menunjukkan perkembangan nilai RMSE pada data pelatihan dan data validasi selama proses training.

<p align="center">
  <img src="grafik_evaluasi_rmse.png" width="700">
</p>

### Analisis Hasil

- Train RMSE mengalami penurunan secara konsisten.
- Validation RMSE relatif stabil.
- Terdapat indikasi overfitting ringan pada model.
- Dropout Layer digunakan untuk membantu meningkatkan kemampuan generalisasi model.

---

## Kesimpulan

Neural Network-Based Recommender System mampu mempelajari hubungan kompleks antara pengguna dan lagu. Meskipun model menunjukkan performa yang baik pada data pelatihan, masih terdapat indikasi overfitting ringan sehingga diperlukan regularisasi dan tuning hyperparameter untuk meningkatkan performa model.
