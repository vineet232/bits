import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights

class ResNet18Temporal(nn.Module):
    def __init__(self, num_classes, pooling="avg", dropout=0.7):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)



    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)

        if self.pooling == "max":
            feats, _ = feats.max(dim=1)
        else:
            feats = feats.mean(dim=1)

        feats = self.dropout(feats)
        out = self.fc(feats)
        return out
    

    ######################################........3D CNN model.......######################################

 

class ResNet3D(nn.Module):
    def __init__(self, num_classes, dropout=0.5, freeze_backbone=True):
        super().__init__()

        base = r3d_18(weights=R3D_18_Weights.DEFAULT)

        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False   # ✅ freeze backbone if false dont freeze if true

        in_features = base.fc.in_features
        base.fc = nn.Identity()   # remove classifier

        self.backbone = base
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x already: (B, C, T, H, W)
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

