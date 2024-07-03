# ImageTextDataset `dataset.py` 

## Deskripsi
`ImageTextDataset` adalah sebuah kelas yang mengimplementasikan `torch.utils.data.Dataset` untuk memuat dataset gambar dan teks. Dataset ini dirancang untuk mempermudah proses pelatihan model pembelajaran mesin yang memerlukan input berupa gambar dan anotasi teks.

## Kegunaan
Kelas ini berguna untuk:
- Memuat gambar dari sebuah folder.
- Memuat anotasi teks dari sebuah file.
- Mengaplikasikan transformasi pada gambar.
- Mengubah anotasi teks menjadi token menggunakan fungsi yang diberikan.

## Fungsi
Kelas `ImageTextDataset` memiliki beberapa fungsi utama:
- `__init__(self, image_folder, annotation_file, tokenize, transform=None)`: Inisialisasi dataset dengan folder gambar, file anotasi, fungsi tokenisasi, dan transformasi opsional.
- `__len__(self)`: Mengembalikan jumlah total anotasi dalam dataset.
- `__getitem__(self, idx)`: Mengembalikan gambar dan anotasi teks yang telah ditokenisasi berdasarkan indeks yang diberikan.
- `load_annotations(self, annotation_file)`: Memuat anotasi dari file dan mengembalikannya dalam bentuk list.

## Bagaimana Menjalankan
Untuk menggunakan `ImageTextDataset`, ikuti langkah-langkah berikut:

1. **Instalasi Dependensi**:
   Pastikan Anda telah menginstal `torch`, `torchvision`, dan `PIL` (Pillow). Anda dapat menginstalnya menggunakan pip:
   ```bash
   pip install torch torchvision pillow
   ```

2. **Persiapkan Dataset**:
   - Letakkan gambar-gambar Anda dalam sebuah folder.
   - Buat file anotasi teks dengan format:
     ```
     nama_gambar1, anotasi teks 1
     nama_gambar2, anotasi teks 2
     ...
     ```

3. **Inisialisasi Dataset**:
   ```python
   from dataset import ImageTextDataset
   from torchvision import transforms

   # Definisikan transformasi (opsional)
   transform = transforms.Compose([
       transforms.Resize((128, 128)),
       transforms.ToTensor()
   ])

   # Definisikan fungsi tokenisasi
   def tokenize(text):
       return text.split()

   # Inisialisasi dataset
   dataset = ImageTextDataset(image_folder='path/to/images', 
                              annotation_file='path/to/annotations.txt', 
                              tokenize=tokenize, 
                              transform=transform)

   # Contoh penggunaan DataLoader
   dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

   for images, annotations in dataloader:
       print(images.shape, annotations)
   ```

## Kesimpulan
`ImageTextDataset` adalah solusi yang efisien untuk memuat dan memproses dataset yang terdiri dari gambar dan anotasi teks. Dengan menggunakan kelas ini, Anda dapat dengan mudah mengelola dan mempersiapkan data untuk pelatihan model pembelajaran mesin. Kelas ini juga mendukung transformasi gambar dan tokenisasi teks, sehingga mempermudah proses pra-pemrosesan data.

---

# ClipFinetuner `finetune.py` 

## Deskripsi
`ClipFinetuner` adalah sebuah modul yang menggunakan `PyTorch Lightning` untuk melakukan fine-tuning pada model CLIP (Contrastive Language-Image Pre-Training) dari OpenAI. Modul ini dirancang untuk mengoptimalkan model CLIP agar lebih sesuai dengan dataset gambar dan teks yang spesifik.

## Kegunaan
Kelas ini berguna untuk:
- Melakukan fine-tuning pada model CLIP menggunakan dataset gambar dan teks kustom.
- Mengoptimalkan representasi gambar dan teks agar lebih relevan dengan tugas tertentu.
- Memanfaatkan kemampuan `PyTorch Lightning` untuk pelatihan yang efisien dan terdistribusi.

## Fungsi
Kelas `ClipFinetuner` memiliki beberapa fungsi utama:
- `__init__(self, clip_model, config)`: Inisialisasi modul dengan model CLIP dan konfigurasi pelatihan.
- `forward(self, image, text)`: Melakukan forward pass untuk mendapatkan fitur gambar dan teks.
- `training_step(self, batch, batch_idx)`: Melakukan satu langkah pelatihan, menghitung loss, dan mencatatnya.
- `configure_optimizers(self)`: Mengonfigurasi optimizer untuk pelatihan.

## Bagaimana Menjalankan
Untuk menggunakan `ClipFinetuner`, ikuti langkah-langkah berikut:

1. **Instalasi Dependensi**:
   Pastikan Anda telah menginstal `torch`, `lightning`, `torchvision`, `PIL` (Pillow), dan `clip`. Anda dapat menginstalnya menggunakan pip:
   ```bash
   pip install torch lightning torchvision pillow git+https://github.com/openai/CLIP.git
   ```

2. **Persiapkan Dataset**:
   - Letakkan gambar-gambar Anda dalam sebuah folder.
   - Buat file anotasi teks dengan format:
     ```
     nama_gambar1, anotasi teks 1
     nama_gambar2, anotasi teks 2
     ...
     ```

3. **Jalankan Script**:
   ```bash
   python finetune_sweep.py --data path/to/dataset --max_steps 100 --lr 1e-3 --batch_size 2
   ```

   Parameter yang dapat disesuaikan:
   - `--data`: Path ke folder dataset.
   - `--max_steps`: Jumlah langkah pelatihan maksimum.
   - `--lr`: Learning rate untuk optimizer.
   - `--batch_size`: Ukuran batch untuk DataLoader.

## Kesimpulan
`ClipFinetuner` adalah solusi yang efisien untuk melakukan fine-tuning pada model CLIP menggunakan dataset gambar dan teks kustom. Dengan menggunakan modul ini, Anda dapat mengoptimalkan model CLIP agar lebih relevan dengan tugas spesifik Anda. Modul ini juga memanfaatkan kemampuan `PyTorch Lightning` untuk pelatihan yang efisien dan terdistribusi.


---

# Grid Search untuk Fine-Tuning CLIP `run_grid_search.py`

## Deskripsi
`run_grid_search.py` adalah sebuah script yang menggunakan `Lightning SDK` untuk melakukan grid search pada hyperparameter pelatihan model CLIP. Grid search ini mencakup berbagai kombinasi learning rate dan ukuran batch untuk menemukan konfigurasi terbaik dalam fine-tuning model CLIP.

## Kegunaan
Script ini berguna untuk:
- Melakukan eksplorasi hyperparameter secara otomatis.
- Menjalankan beberapa eksperimen pelatihan secara paralel.
- Mengoptimalkan proses fine-tuning model CLIP dengan berbagai kombinasi parameter.

## Fungsi
Script `run_grid_search.py` memiliki beberapa fungsi utama:
- `Studio()`: Menginisialisasi referensi ke studio saat ini.
- `install_plugin('jobs')`: Menginstal plugin jobs untuk menjalankan eksperimen.
- `run(cmd, machine, name)`: Menjalankan eksperimen dengan perintah yang diberikan pada mesin yang ditentukan.

## Bagaimana Menjalankan
Untuk menggunakan script `run_grid_search.py`, ikuti langkah-langkah berikut:

1. **Instalasi Dependensi**:
   Pastikan Anda telah menginstal `lightning_sdk`. Anda dapat menginstalnya menggunakan pip:
   ```bash
   pip install lightning_sdk
   ```

2. **Persiapkan Script Fine-Tuning**:
   Pastikan Anda memiliki script `finetune_sweep.py` yang akan digunakan untuk fine-tuning model CLIP.

3. **Jalankan Script Grid Search**:
   Jalankan script `run_grid_search.py` untuk memulai grid search:
   ```bash
   python run_grid_search.py
   ```

   Script ini akan menjalankan beberapa eksperimen dengan berbagai kombinasi learning rate dan ukuran batch pada GPU A10G.

## Kesimpulan
`run_grid_search.py` adalah solusi yang efisien untuk melakukan grid search pada hyperparameter pelatihan model CLIP. Dengan menggunakan script ini, Anda dapat dengan mudah mengeksplorasi berbagai kombinasi parameter dan menemukan konfigurasi terbaik untuk fine-tuning model CLIP. Script ini juga memanfaatkan kemampuan `Lightning SDK` untuk menjalankan eksperimen secara paralel dan terdistribusi.

---

# Random Search untuk Fine-Tuning CLIP `run_random_search.py`

## Deskripsi
`run_random_search.py` adalah sebuah script yang menggunakan `Lightning SDK` untuk melakukan random search pada hyperparameter pelatihan model CLIP. Random search ini memilih beberapa kombinasi learning rate dan ukuran batch secara acak untuk menemukan konfigurasi terbaik dalam fine-tuning model CLIP.

## Kegunaan
Script ini berguna untuk:
- Melakukan eksplorasi hyperparameter secara acak.
- Menjalankan beberapa eksperimen pelatihan secara paralel.
- Mengoptimalkan proses fine-tuning model CLIP dengan berbagai kombinasi parameter yang dipilih secara acak.

## Fungsi
Script `run_random_search.py` memiliki beberapa fungsi utama:
- `Studio()`: Menginisialisasi referensi ke studio saat ini.
- `install_plugin('jobs')`: Menginstal plugin jobs untuk menjalankan eksperimen.
- `run(cmd, machine, name)`: Menjalankan eksperimen dengan perintah yang diberikan pada mesin yang ditentukan.

## Bagaimana Menjalankan
Untuk menggunakan script `run_random_search.py`, ikuti langkah-langkah berikut:

1. **Instalasi Dependensi**:
   Pastikan Anda telah menginstal `lightning_sdk`. Anda dapat menginstalnya menggunakan pip:
   ```bash
   pip install lightning_sdk
   ```

2. **Persiapkan Script Fine-Tuning**:
   Pastikan Anda memiliki script `finetune_sweep.py` yang akan digunakan untuk fine-tuning model CLIP.

3. **Jalankan Script Random Search**:
   Jalankan script `run_random_search.py` untuk memulai random search:
   ```bash
   python run_random_search.py
   ```

   Script ini akan menjalankan beberapa eksperimen dengan berbagai kombinasi learning rate dan ukuran batch pada GPU A10G.

## Kesimpulan
`run_random_search.py` adalah solusi yang efisien untuk melakukan random search pada hyperparameter pelatihan model CLIP. Dengan menggunakan script ini, Anda dapat dengan mudah mengeksplorasi berbagai kombinasi parameter yang dipilih secara acak dan menemukan konfigurasi terbaik untuk fine-tuning model CLIP. Script ini juga memanfaatkan kemampuan `Lightning SDK` untuk menjalankan eksperimen secara paralel dan terdistribusi.
