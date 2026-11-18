"""Utilidades compartidas: semilla, lectura de imágenes, transformaciones, dataset y predicción."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path) -> Image.Image:
    """Lee PNG/JPG o DICOM y devuelve una imagen RGB (escala de grises replicada en 3 canales)."""
    path = str(path)
    if path.lower().endswith(".dcm"):
        import pydicom
        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.open(path)
        img.load()
    # Los recortes son en escala de grises; los modelos ImageNet esperan 3 canales
    return img.convert("L").convert("RGB")


def build_transform(img_size: int, train: bool, aug: bool):
    """Aumentos SOLO en entrenamiento y solo si aug=True. Validación/prueba: sin aumentos."""
    steps = [T.Resize((img_size, img_size))]
    if train and aug:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class LesionDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.transform(load_image(self.paths[i]))
        return x, torch.tensor(self.labels[i])


@torch.no_grad()
def predict_scores(model, loader, device):
    """Devuelve (etiquetas reales, puntuaciones sigmoid) para todo el loader."""
    model.eval()
    ys, ss = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        ss.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ss)


def resolve(path, base: Path):
    p = Path(path)
    return p if p.is_absolute() else base / p
