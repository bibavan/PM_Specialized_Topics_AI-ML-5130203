import torch.nn as nn
from torchvision import models as tmodels
from django.db import models

class ImagePrediction(models.Model):
    image = models.ImageField(upload_to='images/')  # Загрузка изображений в папку 'images/'
    predictions = models.TextField()  # JSON-строка с предсказаниями
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Дата и время загрузки

    def __str__(self):
        return f"Prediction uploaded at {self.uploaded_at}"


class ResNet50Custom(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Custom, self).__init__()

        self.backbone = tmodels.resnet50(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)