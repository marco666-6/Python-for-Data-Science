# Email Spam Detection
## Author : [M Faisal Qureshi](https://github.com/MFaisalQureshi)
## Python for Data Science
### Review dan Modify proyek "[Email Spam Detection](https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy)" authored by https://github.com/MFaisalQureshi
### Mentee assignment from IBM Advance Al @ Infinite Learning Python for Data science project (review code)
#### Mentee Info
##### Name: Marco Philips Sirait
##### Program: IBM Advance AI with Infinite Learning (MSIB)
## LINK SITASI PROYEK & DATASET
---
https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy
---
Email spam detection adalah proses identifikasi email yang tidak diinginkan atau spam menggunakan teknik klasifikasi. Metode klasifikasi Naive Bayes adalah salah satu pendekatan yang umum digunakan untuk masalah ini. Saat digunakan dalam konteks pembelajaran mesin atau deep learning, model Naive Bayes dapat diterapkan untuk membuat sistem yang dapat memprediksi apakah sebuah email merupakan spam atau bukan.

Berikut adalah penjelasan lengkap tentang bagaimana melakukan email spam detection menggunakan teknik klasifikasi Naive Bayes dengan menggunakan model dari library scikit-learn (sklearn) dan menggunakan pipeline:

### 1. Persiapan Data
Sebelumnya, data email perlu dipersiapkan. Ini melibatkan pengumpulan dataset yang berisi email dan label yang menandakan apakah email tersebut spam atau bukan. Dataset kemudian dibagi menjadi dua bagian: satu untuk pelatihan (train set) dan satu lagi untuk pengujian (test set).

### 2. Pra-pemrosesan Data
Pada tahap ini, data email perlu diproses sebelum dapat digunakan untuk pelatihan model. Pra-pemrosesan ini bisa mencakup langkah-langkah seperti:
- Penghapusan karakter khusus, tanda baca, dan kata-kata yang tidak relevan.
- Tokenisasi: Memecah teks email menjadi kata-kata individu atau token.
- Mengubah semua teks menjadi huruf kecil agar konsisten.
- Menghilangkan stopwords (kata-kata umum yang tidak memberikan banyak informasi seperti "the", "is", "and", dll.).
- Melakukan stemming atau lemmatization untuk mengubah kata-kata menjadi bentuk dasarnya.

### 3. Pembuatan Pipeline
Pipeline adalah cara untuk mengatur serangkaian langkah yang diperlukan dalam proses pembuatan model. Dalam kasus ini, pipeline akan mencakup pra-pemrosesan data dan pelatihan model Naive Bayes dalam satu alur.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Membuat pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Konversi teks ke vektor TF-IDF
    ('classifier', MultinomialNB())  # Model klasifikasi Naive Bayes
])
```

Dalam contoh di atas, pipeline terdiri dari dua langkah: pertama, vektorisasi teks menggunakan TF-IDF, dan kedua, klasifikasi menggunakan model Naive Bayes.

### 4. Pelatihan Model
Setelah pipeline dibuat, langkah berikutnya adalah melatih model menggunakan data pelatihan.

```python
pipeline.fit(X_train, y_train)
```

### 5. Evaluasi Model
Setelah model dilatih, langkah terakhir adalah mengevaluasi kinerja model menggunakan data pengujian.

```python
from sklearn.metrics import accuracy_score, classification_report

# Prediksi menggunakan data pengujian
predictions = pipeline.predict(X_test)

# Evaluasi kinerja model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
```

### 6. Penyimpanan Model (Opsional)
Jika model memiliki kinerja yang baik, Anda dapat menyimpannya untuk digunakan di masa depan tanpa harus melatih ulang.

```python
import joblib

# Menyimpan model
joblib.dump(pipeline, 'spam_detection_model.pkl')

# Menggunakan model yang disimpan
loaded_model = joblib.load('spam_detection_model.pkl')
```

Dengan menggunakan pipeline dan model klasifikasi Naive Bayes, Anda dapat membangun sistem deteksi email spam yang efektif dan mudah digunakan. Penting untuk diingat bahwa pra-pemrosesan data yang tepat dan pemilihan fitur yang baik dapat sangat mempengaruhi kinerja model.
