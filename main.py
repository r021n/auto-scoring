import joblib
import numpy as np
from keras.models import load_model

# Muat model
model = load_model('model.h5')

# Muat CountVectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Misalkan kita memiliki teks baru
teks_baru = [str(input("masukan jawabanmu: \n"))]

# Ubah teks baru menjadi vektor
teks_baru_vec = vectorizer.transform(teks_baru)

# Gunakan model untuk memprediksi
prediksi = model.predict(teks_baru_vec)

# Temukan indeks dengan probabilitas tertinggi
kelas_prediksi = np.argmax(prediksi)

print(f'Nilaimu adalah: {kelas_prediksi}')
