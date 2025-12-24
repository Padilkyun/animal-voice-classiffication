# Animal Voice Classification (ANN + MFCC) 🐾

Project ini melakukan klasifikasi suara hewan menggunakan Artificial Neural Network (ANN). Audio tidak diproses sebagai waveform mentah, melainkan diubah menjadi fitur numerik (MFCC + delta + delta²) lalu diklasifikasikan oleh ANN. Project terdiri dari dua bagian:

1) Training & evaluasi model (Google Colab)  
2) Aplikasi inference berbasis Streamlit (local)

---

## Ringkasan Metode

### Input
File audio (.wav/.mp3/.flac/.ogg/.m4a, dsb.)

### Preprocessing & Feature Engineering
Setiap file audio diproses menjadi vektor fitur berdimensi tetap dengan:
- MFCC (n_mfcc = 40)
- Delta MFCC
- Delta-Delta MFCC
- Ringkasan statistik per koefisien: mean dan standard deviation

Hasil per audio menjadi vektor fitur:
- 40 MFCC mean + 40 MFCC std
- 40 delta mean + 40 delta std
- 40 delta2 mean + 40 delta2 std  
Total = 240 fitur per file audio.

### Model
ANN (Dense Network) dengan struktur umum:
- Dense(256, ReLU) + Dropout
- Dense(128, ReLU) + Dropout
- Dense(num_classes, Softmax)

### Output
Probabilitas kelas dan prediksi label hewan.

---

## Dataset

Dataset diunduh menggunakan `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("rushibalajiputthewad/sound-classification-of-animal-voice")
print(path)
