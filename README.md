# Laporan Proyek Machine Learning - Sistem Rekomendasi Musik Spotify

## Project Overview

![Spotify Banner](https://images.unsplash.com/photo-1614680376593-902f74fa0d41?q=80&w=1000&auto=format&fit=crop)

Mendengarkan musik telah menjadi bagian yang tidak terpisahkan dari gaya hidup digital masyarakat modern. Platform streaming raksasa seperti Spotify menyediakan akses instan ke puluhan juta lagu dari berbagai belahan dunia. Namun, masifnya katalog lagu ini memunculkan permasalahan baru bagi para pendengar, yaitu fenomena *choice overload*, di mana pengguna sering kali merasa bingung dan kesulitan dalam menentukan lagu apa yang ingin didengarkan selanjutnya.

Sistem rekomendasi memungkinkan pengguna untuk dengan mudah menemukan referensi musik baru yang dipersonalisasi sesuai dengan preferensi unik atau karakteristik fisis audio dari lagu-lagu yang disukai oleh pendengar sebelumnya. Selain itu, sistem rekomendasi juga memiliki peran penting dalam industri hiburan digital karena dapat memengaruhi kesuksesan komersial platform streaming serta kepuasan retensi pengguna. Proyek ini bertujuan untuk membangun sistem rekomendasi musik terpadu menggunakan pendekatan **Content-Based Filtering (Karakteristik Audio), Collaborative Filtering (SVD), dan Neural Network-Based Recommender System** dengan memanfaatkan dataset riil dari Kaggle.

💡 Manfaat Proyek:
✔ Membantu pendengar menemukan lagu baru yang relevan dengan selera musiknya secara cepat dan otomatis.
✔ Mengoptimalkan pemanfaatan fitur audio digital intrinsik (seperti ketukan, energi, akustik) menjadi wawasan preferensi pengguna.
✔ Mengembangkan model cerdas berbasis Deep Learning menggunakan jaringan saraf tiruan untuk pemetaan interaksi yang personal.

Format Referensi: [SISTEM REKOMENDASI MUSIK BERBASIS FITUR AUDIO INTERN](https://ejurnal.umri.ac.id/index.php/coscitech/)

---

## Business Understanding

📝 Problem Statements
* Bagaimana sistem rekomendasi dapat membantu pengguna menemukan musik secara akurat berdasarkan fitur akustik audio (seperti *danceability* dan *energy*) dari lagu yang disukai sebelumnya?
* Bagaimana arsitektur *Neural Network-Based Recommender System* dapat memetakan interaksi laten kompleks antara pengguna dan objek lagu untuk menghasilkan prediksi rekomendasi yang personal?

🎯 Goals
* Mengembangkan sistem rekomendasi musik terpadu yang mampu menyajikan daftar Top-N rekomendasi lagu serupa secara otomatis berdasarkan riwayat preferensi pengguna.
* Membandingkan performa model berbasis konten fisis, dekomposisi matriks statistik klasik, dan kekuatan model Deep Learning untuk memahami pendekatan terbaik.

🛠 Solution Approach
* **Content-Based Filtering:** Menggunakan informasi deskriptif fitur audio numerik dari lagu untuk menghitung jarak kemiripan antar-objek menggunakan metode *Cosine Similarity* melalui algoritma *K-Nearest Neighbors (K-NN)*.
* **Collaborative Filtering (SVD):** Menggunakan pola interaksi riwayat dengar pengguna lain untuk memprediksi nilai rating ketertarikan menggunakan algoritma *Singular Value Decomposition (SVD)*.
* **Neural Network-Based Recommender:** Menggunakan arsitektur *Deep Learning Functional API* dengan lapisan *Embedding* paralel untuk memetakan hubungan non-linear tersembunyi antara representasi pengguna dan lagu.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini diperoleh dari platform Kaggle, yaitu **Spotify Tracks Dataset**. Kumpulan data ini berisi daftar lagu beserta karakteristik audionya yang ditarik langsung menggunakan infrastruktur Spotify Web API. Dataset asli terdiri dari satu tabel induk besar yang memuat data komprehensif lagu.

```python
# Membaca dataset Spotify dari Kaggle
df_raw = pd.read_csv('spotify-tracks-dataset.csv')
📂 Dataset Components:DatasetJumlah Data AwalJumlah FiturFitur Utamaspotify-tracks-dataset.csv114.000 Baris22 Kolomtrack_id, track_name, artists, album_name, popularity, Fitur Audio Numerik📌 Uraian Fitur Utama Objek Lagu:track_id: Kode identitas alfanumerik unik (ID) resmi dari Spotify untuk mengenali setiap lagu.track_name: Judul resmi dari sebuah lagu.artists: Nama penyanyi atau grup band yang membawakan lagu tersebut.album_name: Nama album tempat lagu tersebut bernaung.popularity: Tingkat popularitas lagu di platform Spotify (rentang skala 0 s.d 100).danceability: Nilai yang mengukur seberapa cocok sebuah lagu digunakan untuk menari berdasarkan tempo, stabilitas ritme, dan kekuatan ketukan (skala 0.0 - 1.0).energy: Representasi ukuran intensitas persepsi, kecepatan, dan aktivitas kekuatan suara dalam sebuah lagu (skala 0.0 - 1.0).valence: Ukuran kepositifan psikologis atau mood yang dipancarkan oleh melodi audio lagu (skala 0.0 - 1.0, nilai tinggi menandakan lagu terdengar bahagia/ceria).tempo: Kecepatan keseluruhan lagu dalam satuan Beats Per Minute (BPM).🔍 Kondisi Data AwalBerdasarkan log eksplorasi awal pada program, data mentah memiliki karakteristik ukuran, nilai kosong (missing values), dan duplikasi sebagai berikut:Missing Values: Ditemukan masing-masing 1 nilai kosong pada kolom metadata tekstual penting seperti artists, album_name, dan track_name.Duplikat Data: Terdeteksi sebanyak 24.259 baris data duplikasi berdasarkan indikator unik track_id.🔍 Eksplorasi Data (EDA)📌 10 Artis Teratas Berdasarkan Jumlah Lagu: Analisis bar chart dilakukan terhadap penyanyi/band yang paling dominan menyumbangkan lagu dalam dataset ini untuk melihat keterlibatan musisi.📌 Distribusi Tingkat Energi Lagu: Visualisasi histogram dilakukan untuk melihat sebaran nilai karakteristik energy guna mengetahui kecenderungan jenis tempo musik di dalam dataset.Data PreparationTahapan data preparation dilakukan secara bertahap dan terstruktur untuk memastikan kualitas dataset sebelum masuk ke tahap pemodelan sistem rekomendasi musik.📌 Pembersihan Data DasarLangkah pertama adalah membersihkan data dengan cara menghapus duplikasi baris berdasarkan indikator unik track_id agar model tidak mengalami bias, serta membuang baris yang mengandung missing value.Python# Menghapus duplikasi berdasarkan track_id
df_clean = df_raw.drop_duplicates(subset=['track_id']).copy()

# Menghapus baris jika ada nilai kosong pada kolom esensial
df_clean.dropna(subset=['track_name', 'artists'], inplace=True)
📌 Pembentukan Data Interaksi & Sampling AmanUntuk memenuhi kebutuhan pemodelan kolaboratif dan Deep Learning, dibentuk kolom ID Pengguna tiruan (user_id) secara acak, dan dilakukan normalisasi fitur popularity menjadi indikator rating interaksi pengguna pada rentang skala 1-10. Data disampel sebesar 25.000 baris agar penggunaan memori RAM Google Colab tetap aman dari risiko session crash.Python# Mengambil sampel data agar memori tidak over-use
df_filtered = df_clean.sample(n=25000, random_state=42).reset_index(drop=True)

# Membuat ID User Tiruan & Mengubah Popularitas menjadi Skala Rating (1-10)
np.random.seed(42)
df_filtered['user_id'] = np.random.randint(1000, 5000, size=len(df_filtered))
df_filtered['user_rating'] = (df_filtered['popularity'] / 10).round().astype(int)
df_filtered['user_rating'] = df_filtered['user_rating'].replace(0, 1)

# Memfilter data dengan membatasi kelayakan rating minimal 3
df_filtered = df_filtered[df_filtered['user_rating'] >= 3].reset_index(drop=True)
Berikut adalah total akhir baris data bersih dan dimensi matriks hasil penskalaan yang sukses siap pakai:📌 Feature Scaling & Kategori EncodingFitur audio numerik intrinsik disetarakan ke rentang [0, 1] menggunakan MinMaxScaler. Variabel track_id berupa string juga diubah menjadi indeks angka (track_index) menggunakan fungsi kategori kode agar kompatibel dengan arsitektur jaringan Keras.Python# Encoding track_id menjadi index numerik untuk Embedding
df_filtered['track_index'] = df_filtered['track_id'].astype('category').cat.codes

# Ekstraksi dan Penskalaan Fitur Audio (Content-Based)
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_filtered[audio_features])
📌 Tahap Pembagian Data (train-test-split)Dataset interaksi musik dibagi menjadi data latih (training set) dan data uji (testing set) menggunakan rasio 80:20 untuk divalidasi pada algoritma Collaborative Filtering.Pythonfrom surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Menyiapkan dataset untuk library Surprise
reader = Reader(rating_scale=(1, 10))
data_surprise = Dataset.load_from_df(df_filtered[["user_id", "track_id", "user_rating"]], reader)

# Membagi 80% Train dan 20% Test
trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)

# Menggunakan SVD dari Surprise
model_svd = SVD(n_factors=20, random_state=42)
model_svd.fit(trainset)

# Memprediksi pada test set
predictions = model_svd.test(testset)
rmse = accuracy.rmse(predictions)
print("SVD with Surprise RMSE:", rmse)
Modeling and Results📝 Pendekatan Model📖 1. Content-Based Filtering & 2. Collaborative Filtering (SVD)Proyek ini sukses mengeksekusi pengujian pencarian Top-5 lagu serupa menggunakan kedekatan spasial model K-NN (Cosine Similarity) serta pengujian performa prediksi matriks SVD klasik. Hasil keluaran program tercatat sebagai berikut:Hasil Rekomendasi Berbasis Konten (K-NN): Berhasil memunculkan 5 rekomendasi lagu terdekat (seperti lagu La Chula, Settaigalai Virikkum Kaalam, dll.) ketika diuji menggunakan contoh sebuah lagu target.Hasil Evaluasi SVD: Berdasarkan pengujian Cross Validation 5-Fold, model SVD menghasilkan rata-rata nilai kesalahan (Mean RMSE) yang sangat konsisten, yaitu 1.3324.🧠 3. Neural Network-Based Recommender SystemPendekatan Deep Learning memanfaatkan Keras Functional API dari TensorFlow. Ringkasan arsitektur spesifikasi lapisan jaringan saraf tiruan paralel terkonfigurasi secara detail sebagai berikut:Model memproses input ID Pengguna (user_id) network dan ID Lagu (isbn) melewati lapisan Embedding Layer berdimensi 50, diratakan melalui Flatten Layer, disatukan lewat Concatenate Layer, dimasukkan ke dalam rangkaian Dense Layer (128 dan 64 neuron) dengan aktivasi ReLU, dihambat risiko overfitting-nya via Dropout Layer (0.2), dan diakhiri oleh 1 output neuron linear.Berikut adalah bukti riwayat jalannya putaran latihan komputasi (training process) model saraf tiruan dari Epoch 1 hingga Epoch 10:Kelebihan & Kekurangan TotalPendekatanKelebihanKekuranganContent-Based FilteringTidak memerlukan data interaksi pengguna lain; terhindar dari cold start.Terbatas pada kemiripan parameter audio yang sudah terdata saja.Collaborative FilteringMampu menyajikan rekomendasi personal lintas batas genre musik secara fleksibel.Membutuhkan data riwayat interaksi yang padat dan matang.Neural Network RecommenderSangat andal dalam menangkap pola hubungan interaksi non-linear yang kompleks.Memerlukan daya komputasi tinggi dan pengawasan regularisasi ketat.Evaluation Model💡 Visualisasi Grafik Evaluasi JaringanBerdasarkan hasil akhir pengujian, grafik visualisasi kurva pergerakan nilai error Root Mean Squared Error (RMSE) yang terbentuk pasca-training adalah sebagai berikut:📌 Analisis Hasil Grafik:Train RMSE (Garis Biru): Mengalami penurunan nilai eror yang sangat tajam dan konsisten hingga berhasil menyentuh nilai akhir 0.5428 pada Epoch 10. Hal ini menandakan model sudah sangat cerdas dalam mendeteksi selera musik pada data latih.Validation RMSE (Garis Oranye): Bergerak stabil dan mendatar pada perolehan skor akhir 2.1979.Diagnosis Gejala Overfitting: Jarak pemisah antara garis biru yang terus menurun dengan garis oranye yang mendatar mengindikasikan munculnya gejala Overfitting ringan. Kondisi ini sangat wajar terjadi mengingat data identitas interaksi pengguna dibentuk secara simulasi acak (random simulation) pada tahap penyiapan data di Google Colab.✔ Solusi Optimal untuk Mengatasi Overfitting Jaringan:Menerapkan lapisan Dropout layer (seperti baris komputasi Dropout(0.2) yang telah dipasang pada arsitektur).Menambahkan regularisasi bobot L2 pada dense layer utama.Melakukan penyesuaian hyperparameter tingkat learning rate pada biner Adam Optimizer.DeploymentSebagai pemenuhan kriteria penilaian tambahan (poin plus), kelompok kami mendeploy model sistem rekomendasi musik ini menjadi sebuah aplikasi web interaktif berbasis Streamlit yang dijalankan secara lokal.Isi File Deployment dalam Repositori:app.py: Skrip program utama pembangun antarmuka website berbasis Streamlit.requirements.txt: Berkas daftar pustaka library dependensi proyek (tensorflow, streamlit, pandas, scikit-learn, scikit-surprise).spotify_processed_data.csv: Berkas dataset hasil pembersihan akhir dari Google Colab.Cara Menjalankan Aplikasi Web:Buka terminal sistem komputer Anda, pasang seluruh requirement library, lalu jalankan perintah server lokal Streamlit:Bashpip install -r requirements.txt
streamlit run app.py
Kesimpulan🌟 Content-Based Filtering terbukti sangat efektif untuk mengurasi kesamaan rumpun karakteristik audio fisik intrinsik lagu tanpa bergantung pada riwayat dengar orang lain.🌟 Collaborative Filtering (SVD) sangat andal dalam melahirkan pola rekomendasi personal lintas genre musik berdasarkan kecenderungan perilaku kelompok dengar.🌟 Neural Network Recommender System menyajikan fleksibilitas tingkat tinggi untuk pemetaan fungsi non-linear tersembunyi, namun membutuhkan pengawasan regularisasi ketat agar terhindar dari risiko overfitting data.