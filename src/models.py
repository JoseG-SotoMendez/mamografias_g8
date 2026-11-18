"""
Modelos:
  smallcnn        línea base: CNN pequeña entrenada desde cero (pocas capas)
  resnet18        transferencia de aprendizaje (pesos ImageNet)
  efficientnet_b0 transferencia de aprendizaje (pesos ImageNet)

Todos devuelven UN logit (salida de tamaño 1); la puntuación es sigmoid(logit).
"""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        head = m.fc
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        head = m.classifier
    else:
        raise ValueError(f"modelo desconocido: {name}")
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
