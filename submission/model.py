# Submission Studio 2 - Klasifikasi Sampah (IF25-40401)
# File ini dihasilkan otomatis oleh notebook pelatihan. get_model() hanya membangun arsitektur;
# bobot dimuat oleh evaluate.py dari model.pth. Urutan kelas (alfabetis, = ImageFolder):
# 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 5
ARCHS = ['convnext_tiny']          # nama arsitektur timm tiap anggota ensemble
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TTA_FLIP = True


class Ensemble(nn.Module):
    def __init__(self, archs, tta_flip):
        super().__init__()
        self.members = nn.ModuleList(
            [timm.create_model(a, pretrained=False, num_classes=NUM_CLASSES) for a in archs])
        self.tta_flip = tta_flip

    def forward(self, x):
        total, n = 0.0, 0
        for m in self.members:
            views = (x, torch.flip(x, dims=[-1])) if self.tta_flip else (x,)
            for v in views:
                total = total + m(v).float().softmax(dim=-1)
                n += 1
        return (total / n).clamp_min(1e-8).log()      # log-probabilitas [batch, 5]; argmax = prediksi


def get_model():
    return Ensemble(ARCHS, TTA_FLIP)


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
