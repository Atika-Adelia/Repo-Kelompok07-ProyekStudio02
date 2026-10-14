# Studio 2 — Klasifikasi Sampah · Kelompok 07

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No |          Nama Lengkap         |   NIM     |  Kontibusi Utama |
|----|-------------------------------|-----------|  -----------                |
| 1  | Hildyah Maretasya Araffad     |123140151  |   On Process               |
| 2  | Firman H Gultom               |123140171  |   On Process              |
| 3  | Atika Adelia                  |123140172  |   On Process              |

## Ringkasan Pendekatan
 
- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k` dari `timm`, 27,8 juta parameter), pretrained ImageNet-22k yang di-fine-tune ke ImageNet-1k. Head klasifikasi diganti menjadi 5 kelas (`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`). Submission berupa satu model; `model.py` memakai kelas `Ensemble` dengan satu anggota.
- **Strategi pelatihan**:
  - Fine-tuning penuh (tanpa freezing), 15 epoch, batch 32, mixed precision fp16 pada GPU Tesla T4 (± 6,3 menit).
  - Optimizer AdamW dengan learning rate berbeda: backbone `1e-4`, head `1e-3`. Weight decay 0,05 (tidak dikenakan pada bias dan parameter 1-D). Gradient clipping 1,0.
  - Scheduler: warmup linear 1 epoch, lalu cosine decay hingga 1% dari LR awal (diperbarui tiap step).
  - Regularisasi: label smoothing 0,1, drop path 0,1, dropout head 0,2, Mixup (α=0,2) / CutMix (α=1,0) dengan peluang 0,5 yang dimatikan pada 3 epoch terakhir.
  - EMA bobot (decay 0,995); bobot EMA pada epoch terakhir yang disimpan (`select='last'`, tidak memilih checkpoint berdasarkan validasi).
  - Tanpa penyeimbangan kelas (rasio kelas terbesar/terkecil hanya 1,84).
  - Augmentasi: `RandomResizedCrop` (scale 0,35–1,0), `RandomHorizontalFlip`, `TrivialAugmentWide`, `RandomErasing` (p=0,2).
- **Data tambahan**: tidak ada. Hanya dataset resmi (2.753 gambar, 5 kelas). Pemeriksaan integritas menemukan 0 gambar rusak, 0 duplikat persis (md5), dan 0 label konflik.
- **Ukuran input & preprocessing**:
  - Input **224×224**, normalisasi mean/std dari `pretrained_cfg` backbone (ImageNet).
  - Training: gambar di-cache dengan sisi terpanjang maks. 512 px (JPEG kualitas 95), lalu melewati pipeline augmentasi di atas.
  - Inferensi (`get_transform()`): `Resize((224, 224))` bicubic → `ToTensor` → `Normalize`.
  - **TTA**: rata-rata probabilitas dari gambar asli dan horizontal flip.
  - 
## Hasil
 
| Putaran | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) |    0,9767 |   |

 
Nilai lokal diukur dengan `evaluate.py` pada `my_val/` (413 gambar, split stratified 85/15, `seed=42`), dengan pengaturan submission (224 px + TTA flip).
 
Detail per kelas (`evaluate.py`, `my_val/`):
 
| Kelas         | F1     | Jumlah gambar |
|---------------|--------|---------------|
| botol_plastik | 0,9873 | 78            |
| gelas_plastik | 0,9806 | 78            |
| kertas_kardus | 0,9558 | 55            |
| saset_kemasan | 0,9851 | 101           |
| styrofoam     | 0,9749 | 101           |
| **Macro-F1**  | **0,9767** |           |
 
Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
 
- **Split data**: 2.340 gambar untuk train dan 413 untuk validasi lokal (stratified, proporsi kelas terjaga).
- **Proses training**: Macro-F1 validasi (EMA) naik dari 0,920 pada epoch 1 menjadi 0,972 pada epoch 15 (puncak 0,9737 pada epoch 13–14). Bobot EMA cenderung lebih stabil daripada bobot biasa di epoch-epoch akhir (0,9719 vs 0,9697 pada epoch 15).
- **Resolusi uji dan TTA** (pada model yang sama):
  | Resolusi | Tanpa TTA | TTA flip |
  |----------|-----------|----------|
  | 224      | 0,9719    | 0,9745   |
  | 256      | 0,9719    | 0,9742   |
  | 288      | 0,9757    | 0,9757   |
- TTA flip memberi kenaikan kecil di 224 px. Resolusi 288 px sedikit lebih tinggi, tetapi selisihnya (+0,0012) di bawah ambang pemilihan kami (0,003) dan biaya inferensi di CPU lebih besar, sehingga dipilih **224 px + TTA**.
- **Selisih notebook vs `evaluate.py`**: 0,9745 (notebook, gambar cache dan bobot fp32) vs 0,9767 (`evaluate.py`, gambar asli dan bobot fp16 dari `model.pth`); selisih +0,0023.
- **Analisis kesalahan**: kelas terlemah adalah `kertas_kardus` (F1 0,9558; kelas dengan data paling sedikit, 365 gambar). Kesalahan terbanyak adalah `styrofoam` yang diprediksi `kertas_kardus` (3 gambar).
- **Anggaran waktu inferensi**: pada CPU 2 core, praproses ± 15 ms dan forward ± 294 ms per gambar, setara ± 3,8 menit untuk 740 gambar (batas 15 menit).
- **Belum dicoba**: data tambahan, ensemble beberapa backbone (pipeline sudah mendukung, tetapi submission saat ini satu model), dan sampler penyeimbang kelas.
## Cara Reproduksi
 
Pelatihan dilakukan lewat notebook `Studio2_Kelompok07_RB.ipynb` (dijalankan di Google Colab, GPU Tesla T4; torch 2.11.0, timm 1.0.30).
 
```bash
pip install -r submission/requirements.txt        # timm==1.0.30
pip install torch torchvision scikit-learn pandas matplotlib pillow tqdm
```
 
1. Letakkan dataset resmi di Google Drive pada `MyDrive/dataset_studio2/<nama_kelas>/` (atau ubah `CFG.data_root`). Jika dijalankan lokal, letakkan di `./train/<nama_kelas>/`.
2. Jalankan seluruh sel notebook secara berurutan. Notebook akan melatih model, membuat `my_val/`, memilih pengaturan evaluasi, mengekspor `submission/` (`model.py`, `model.pth`, `requirements.txt`), mengunduh `evaluate.py`, dan menjalankan self-test.
3. Evaluasi ulang secara mandiri:
```bash
python evaluate.py --submission ./submission --data ./my_val
```
 
Catatan: `seed=42` dipakai di seluruh pipeline, tetapi `cudnn.benchmark=True` sehingga hasil antar-run bisa berbeda tipis.
 
## Struktur Repositori
 
```
Repo-Kelompok07-ProyekStudio02
├── 
├── submission/
│  ├──model.py                    # get_model() dan get_transform()
│  ├──model.pth                   # state_dict model terbaik 
│  ├──requirements.txt            # timm==1.0.30
├──evaluate.py                    # salinan skrip evaluasi resmi
├──Studio2_Kelompok07_RB.ipynb    # notebook pelatihan, ekspor submission, dan self-test

```
