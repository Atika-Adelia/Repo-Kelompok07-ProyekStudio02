# Studio 2 — Klasifikasi Sampah · Kelompok 07

<p align="center">
<img src="https://img.shields.io/badge/Google_Colab-20232A?style=for-the-badge&logo=googlecolab&logoColor=F9AB00" alt="Google Colab"/>
<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
</p>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No |          Nama Lengkap         |   NIM     |  Kontibusi Utama |
|----|-------------------------------|-----------|  -----------     |
| 1  | Hildyah Maretasya Araffad     |123140151  |   Modeling - Training     |
| 2  | Firman H Gultom               |123140171  |   Evaluasi - Analisis      |
| 3  | Atika Adelia                  |123140172  |   Preprocessing - Split & Pipeline data    |

## Ringkasan Pendekatan
 
- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k` dari `timm`, 27,8 juta parameter), pretrained ImageNet-22k yang di-fine-tune ke ImageNet-1k. Head klasifikasi diganti menjadi 5 kelas (`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`). Submission berupa satu model; `model.py` memakai kelas `Ensemble` dengan satu anggota. Bobot di `model.pth` disimpan dalam fp16 (± 53,1 MB).
- **Strategi pelatihan**:
  - Fine-tuning penuh (tanpa freezing), 15 epoch, batch 32, mixed precision fp16 pada GPU Tesla T4 (± 5,4 menit).
  - Optimizer AdamW dengan learning rate berbeda: backbone `1e-4`, head `1e-3`. Weight decay 0,05 (tidak dikenakan pada bias dan parameter 1-D). Gradient clipping 1,0.
  - Scheduler: warmup linear 1 epoch, lalu cosine decay hingga 1% dari LR awal (diperbarui tiap step).
  - Regularisasi: label smoothing 0,1, drop path 0,1, dropout head 0,2, Mixup (α=0,2) / CutMix (α=1,0) dengan peluang 0,5 yang dimatikan pada 3 epoch terakhir.
  - EMA bobot (decay 0,995, dengan warm-up decay di awal training); bobot EMA pada epoch terakhir (epoch 15) yang disimpan (`select='last'`, tidak memilih checkpoint berdasarkan validasi).
  - Tanpa penyeimbangan kelas (rasio kelas terbesar/terkecil hanya 1,84).
  - Augmentasi: `RandomResizedCrop` (scale 0,35–1,0), `RandomHorizontalFlip`, `TrivialAugmentWide`, `RandomErasing` (p=0,2).
- **Data tambahan**: tidak ada. Hanya dataset resmi (2.753 gambar, 5 kelas: 523 / 521 / 365 / 673 / 671 untuk `botol_plastik` / `gelas_plastik` / `kertas_kardus` / `saset_kemasan` / `styrofoam`). Pemeriksaan integritas menemukan 0 gambar rusak, 0 duplikat persis (md5), dan 0 label konflik; tidak ada gambar yang dikeluarkan dari training.
- **Ukuran input & preprocessing**:
  - Training pada **224×224**; inferensi pada **288×288**. Resolusi uji dipilih dari hasil pindai resolusi + TTA pada validasi lokal (bagian 7.1 notebook); model **tidak** dilatih ulang pada 288 px. Normalisasi mean/std dari `pretrained_cfg` backbone (ImageNet).
  - Training: gambar di-cache dengan sisi terpanjang maks. 512 px (JPEG kualitas 95), lalu melewati pipeline augmentasi di atas.
  - Inferensi (`get_transform()`): `Resize((288, 288))` bicubic → `ToTensor` → `Normalize`. Nilai resolusi ditulis sebagai konstanta di `model.py` (`IMG_SIZE = 288`).
  - **TTA**: tidak dipakai pada submission (`TTA_FLIP = False`). Implementasi rata-rata probabilitas gambar asli dan horizontal flip tetap tersedia di `model.py`, tetapi dimatikan karena pada 288 px justru menurunkan Macro-F1 validasi (0,9759 vs 0,9806) dan menggandakan waktu inferensi.

## Hasil
 
| Putaran | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) |    0,9767 | 0,9654 |
| Preview 2 (21 Okt) |    0,9813 | 0,9705  |
| Preview 3 (25 Okt) |   0,9813  | ongoing  |
 
Nilai lokal diukur dengan `evaluate.py` pada `my_val/` (413 gambar, split stratified 85/15, `seed=42`).
 
Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal) :
 
- **Split data**: 2.340 gambar untuk train dan 413 untuk validasi lokal (stratified, proporsi kelas terjaga). Jumlah per kelas (train / val): `botol_plastik` 445 / 78, `gelas_plastik` 443 / 78, `kertas_kardus` 310 / 55, `saset_kemasan` 572 / 101, `styrofoam` 570 / 101.
- **Proses training** (run `holdout_convnext_tiny_224`, 5,4 menit): Macro-F1 validasi (EMA) naik dari 0,9201 pada epoch 1 menjadi 0,9710 pada epoch 15, dengan puncak 0,9760 pada epoch 11–12. Loss turun dari 1,207 ke 0,444; penurunan tajam di epoch 13 (0,617 → 0,436) bersamaan dengan Mixup/CutMix dimatikan, sehingga tidak sebanding langsung dengan loss epoch sebelumnya. Bobot EMA lebih stabil daripada bobot biasa saat learning rate masih besar (epoch 5–7, mis. 0,9697 vs 0,9503 pada epoch 7), tetapi di epoch-epoch akhir keduanya hampir identik (0,9710 vs 0,9710 pada epoch 15) karena learning rate sudah sangat kecil. Checkpoint yang disimpan adalah epoch 15 (`select='last'`; Macro-F1 0,9710, akurasi 0,9734), bukan puncak validasi; selisihnya 0,005 setara ± 2 gambar.
- **Resolusi uji dan TTA** (pada model yang sama):
  | Resolusi | Tanpa TTA | TTA flip |
  |----------|-----------|----------|
  | 224      | 0,9710    | 0,9761   |
  | 256      | 0,9739    | 0,9743   |
  | 288      | 0,9806    | 0,9759   |
- TTA flip menaikkan Macro-F1 di 224 px (+0,0051), nyaris tidak berpengaruh di 256 px (+0,0004), dan menurunkannya di 288 px (−0,0047). Pengaturan awal (224 px + TTA, 0,9761) hanya diganti bila ada kandidat yang lebih baik minimal 0,003; 288 px tanpa TTA (0,9806, +0,0045) lolos ambang tersebut sehingga dipilih **288 px tanpa TTA**, yang juga lebih hemat karena hanya satu kali forward. Catatan: selisih tersebut setara ± 2 gambar dari 413 dan dipilih pada validasi yang sama, sehingga cenderung optimistis.
- **Selisih notebook vs `evaluate.py`**: 0,9806 (notebook, gambar cache dan bobot fp32) vs 0,9813 (`evaluate.py`, gambar asli dan bobot fp16 dari `model.pth`); selisih +0,0007.
- **Analisis kesalahan**: pada pengaturan akhir (288 px, tanpa TTA) 7 dari 413 gambar salah (akurasi 0,9831). F1 per kelas: `botol_plastik` 1,0000, `gelas_plastik` 0,9806, `kertas_kardus` 0,9558, `saset_kemasan` 0,9901, `styrofoam` 0,9800. Kelas terlemah adalah `kertas_kardus` (kelas dengan data paling sedikit, 365 gambar): 4 dari 7 kesalahan adalah gambar kelas lain yang diprediksi `kertas_kardus` (2 `styrofoam`, 1 `gelas_plastik`, 1 `saset_kemasan`), sehingga precision-nya paling rendah. Kesalahan terbanyak adalah `styrofoam` yang diprediksi `kertas_kardus` (2 gambar). Pada 224 px (sebelum resolusi uji dinaikkan) ada 11 gambar salah dan `kertas_kardus` juga terlemah (F1 0,9464).
- **Anggaran waktu inferensi**: pada CPU 2 core (288 px, tanpa TTA), praproses ± 15 ms dan forward ± 217 ms per gambar, setara ± 2,9 menit untuk 740 gambar (batas 15 menit). Pengukuran memakai bobot acak di Colab, sehingga mesin penilaian bisa lebih lambat.
- **Belum dicoba**: data tambahan. Pada percobaan ini juga belum dicoba melatih pada resolusi di atas 224 px (resolusi 288 px baru dipakai saat inferensi) maupun memilih checkpoint terbaik berdasarkan validasi (`select='best'`).

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
├──submission/
│  ├──model.py                    
│  ├──model.pth                   
│  ├──requirements.txt            
├──README.md
├──evaluate.py                   
├──Studio2_Kelompok07_RB.ipynb   
```
