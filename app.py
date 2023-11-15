# Import library yang diperlukan
import joblib
import numpy as np
from keras.models import load_model

# Muat model
# Jika model Anda adalah model Keras, gunakan kode berikut:
model = load_model('model.h5')

# Jika model Anda adalah model scikit-learn, gunakan kode berikut:
# model = joblib.load('model.pkl')

# Muat CountVectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Misalkan kita memiliki teks baru
teks_baru = [str(input("masukkan jawabanmu : \n"))]

# Ubah teks baru menjadi vektor
teks_baru_vec = vectorizer.transform(teks_baru)

# Gunakan model untuk memprediksi
prediksi = model.predict(teks_baru_vec)

# Jika model Anda adalah model Keras, gunakan kode berikut untuk mendapatkan kelas prediksi:
kelas_prediksi = np.argmax(prediksi)

# Jika model Anda adalah model scikit-learn, prediksi sudah dalam bentuk kelas, jadi Anda tidak perlu menggunakan np.argmax

print(f'nilaimu adalah: {kelas_prediksi}')
