<div align="center">

# 📈 Sistem Prediksi Harga Saham Berbasis Deep Learning
### (Web Crawling & Hybrid BiLSTM-CNN)

[![Python](https://img.shields.io/badge/Language-Python_3-blue.svg)]()
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow%20%7C%20Keras-orange.svg)]()
[![Data Science](https://img.shields.io/badge/Data_Science-Pandas%20%7C%20Scikit--Learn-green.svg)]()

</div>

---

## 📖 Deskripsi Proyek
Proyek ini dikembangkan untuk membangun sistem analitik prediksi harga saham yang komprehensif. Menggabungkan teknik akuisisi data (*Web Crawling*) secara *real-time* dengan arsitektur kecerdasan buatan kelas atas, proyek ini dipecah menjadi dua pilar utama yang saling melengkapi:

### 1. Akuisisi Data & Analisis Awal (*Web Crawling*)
Dibangun menggunakan skrip `crawling.py`, tahap ini bertujuan untuk membentuk pondasi data yang kuat:
- **Sumber Data:** Mengekstrak data *real-time* dari 10 perusahaan teratas di dunia langsung dari situs web **TradingView**.
- **Fitur Ekstraksi:** Nama perusahaan, kode saham, harga saham saat ini, dan *market cap*. Setiap data secara otomatis dicap dengan *timestamp* dan disimpan secara historis di `HargaSaham.csv`.

<div align="center">
  <img src="assets/image_2.jpeg" width="800" alt="Tangkapan Layar CSV Harga Saham">
</div>

- **Pre-Processing:** Membersihkan format angka yang kotor (misalnya sufiks 'M' untuk Juta, 'B' untuk Miliar, dan 'T' untuk Triliun) untuk membentuk `HargaSaham_Rounded.csv`. File bersih ini dioptimalkan agar dapat divisualisasikan menggunakan *dashboard* analitik seperti **Metabase**.

<div align="center">
  <img src="assets/image_3.jpeg" width="800" alt="Metabase Dashboard Visualisasi Saham">
</div>

---

### 2. Prediksi Harga Saham (Hybrid BiLSTM-CNN)
Tahap ini merupakan inti analisis prediktif yang dioperasikan oleh skrip `lstm_trainingV2.py` dan `testing_lstmV2.py`.
Model menggunakan arsitektur gabungan (Hybrid):
- **Convolutional Neural Network (CNN):** Digunakan pada tahap awal arsitektur untuk menambang fitur spasial dan mendeteksi pola lokal dari kumpulan data *time-series* secara efektif.
- **Bidirectional Long Short-Term Memory (Bi-LSTM):** Mengambil fitur yang telah diekstrak oleh lapisan CNN untuk menangkap pola temporal dan dependensi jangka panjang dari kedua arah maju dan mundur (bidirectional).

<div align="center">
  <img src="assets/image_4.jpeg" width="800" alt="Proses Training Epoch CNN-BiLSTM">
</div>

#### **Hasil Prediksi Aktual vs Prediksi Model:**
Grafik di bawah ini menunjukkan seberapa akurat model mengikuti tren pergerakan harga saham Apple, di mana *gap* antara garis aktual dan garis prediksi sangat tipis.
<div align="center">
  <img src="assets/image_5.png" width="800" alt="Grafik Actual vs Predicted">
</div>

#### **Uji Prediksi Masa Depan (*Forecasting*):**
Skrip testing memungkinkan untuk memuat model *pretrained* untuk meramalkan harga saham satu hari di depan.
<div align="center">
  <img src="assets/image_6.jpeg" width="800" alt="Hasil Console Testing Prediksi">
</div>

---

## 🧠 Detail Arsitektur & Pelatihan Model
Keberhasilan pemodelan sangat bergantung pada struktur dan parameter jaringan:
- **Dataset:** Dataset historis komprehensif dari 8 saham terbesar di dunia, memuat fitur: `Open`, `High`, `Low`, `Price`, `Vol.`, dan `Change %`.
- **Normalisasi:** Seluruh fitur numerik diselaraskan ke rentang `[0, 1]` menggunakan `MinMaxScaler`.
- **Sliding Window:** Data sekuensial disusun dengan panjang jendela waktu (sequence length) selama **90 hari** untuk memprediksi harga di **1 hari ke depan**.
- **Kompilasi Model:** Model dilatih menggunakan *optimizer* `Adam` dan dievaluasi dengan fungsi *loss* `Mean Squared Error (MSE)`.
- **Early Stopping:** Mencegah terjadinya *overfitting* selama proses iterasi berulang yang panjang.
- **Penyimpanan:** Model prediksi disimpan dalam format `.h5` dan skala normalisasi disimpan sebagai `.gz`, memungkinkan prediksi luring (*offline*) secara instan tanpa perlu melatih ulang dari nol.

---

## 🚀 Cara Menjalankan Program
1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/ali1140/Prediksi_HargaSaham_using-BILSTM-CNN.git
   cd Prediksi_HargaSaham_using-BILSTM-CNN
   ```
2. **Instalasi Pustaka:**
   Pastikan Anda menginstal TensorFlow, Pandas, Scikit-Learn, dan pustaka pendukung lainnya (sebaiknya di dalam *virtual environment*).
3. **Web Crawling (Opsional):**
   Untuk memperbarui *database* CSV dengan harga saat ini:
   ```bash
   python crawling.py
   ```
4. **Melatih Model AI:**
   Latih model *Hybrid* Anda menggunakan sekumpulan data historis:
   ```bash
   python lstm_trainingV2.py
   ```
5. **Memprediksi Harga Masa Depan:**
   Jalankan sistem prediksi iteratif untuk menebak pergerakan harga satu langkah di depan dengan memanggil model yang sudah disimpan (*pretrained*):
   ```bash
   python testing_lstmV2.py
   ```

---
## 👨‍💻 Kontributor
**Ali Akbar Alhabsyi (ali1140)**  
Eksplorasi di ranah *Big Data* dan *Deep Learning* tingkat mahir. Menghadapi tantangan optimasi algoritma *parsing* data statis web serta *tuning* hyperparameter jaringan saraf sekuensial untuk mencari *sweet spot* akurasi pemodelan temporal.