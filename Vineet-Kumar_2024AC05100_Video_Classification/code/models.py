import torch
import torch.nn as nn

# Pretrained 2D ResNet-18 for frame-level feature extraction
from torchvision.models import resnet18, ResNet18_Weights

# Pretrained 3D ResNet-18 for spatiotemporal feature learning
from torchvision.models.video import r3d_18, R3D_18_Weights


############################################################
#              2D CNN WITH TEMPORAL POOLING
############################################################

class ResNet18Temporal(nn.Module):
    
    # Here the model uses 2D CNN to extract spatial features from each video frame
    # independently, followed by temporal pooling across frames.
    

    def __init__(self, num_classes, pooling="avg", dropout=0.7):
        super().__init__()

        # Load ImageNet-pretrained ResNet-18
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final classification layer to use ResNet as a feature extractor
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        # Allow fine-tuning of the backbone
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        # Temporal pooling strategy: average or max
        self.pooling = pooling

        # Dropout applied after temporal pooling
        self.dropout = nn.Dropout(dropout)

        # Classification layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input tensor shape:
        # x → (batch_size, num_frames, channels, height, width)

        B, T, C, H, W = x.shape

        # Collapse batch and time to process frames independently
        x = x.reshape(B * T, C, H, W)

        # Extract spatial features per frame
        features = self.feature_extractor(x)

        # Remove singleton spatial dimensions
        features = features.squeeze(-1).squeeze(-1)

        # Restore temporal structure
        features = features.view(B, T, -1)

        # Aggregate features across time dimension
        if self.pooling == "max":
            features, _ = torch.max(features, dim=1)
        else:
            features = torch.mean(features, dim=1)

        # Regularization
        features = self.dropout(features)

        # Final prediction
        output = self.fc(features)

        return output


############################################################
#                 3D CNN BASED MODEL
############################################################

class ResNet3D(nn.Module):
    
    # Here the model uses 3D convolutional backbone to jointly learn
    # spatial and temporal representations from video clips.
    

    def __init__(self, num_classes, dropout=0.5, freeze_backbone=True):
        super().__init__()

        # Load Kinetics-pretrained 3D ResNet-18
        backbone_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in backbone_model.parameters():
                param.requires_grad = False

        # Store feature dimensionality before classification
        feature_dim = backbone_model.fc.in_features

        # Remove original classification head
        backbone_model.fc = nn.Identity()

        self.backbone = backbone_model

        # Dropout for improved generalization
        self.dropout = nn.Dropout(dropout)

        # Task-specific classifier
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Input tensor shape:
        # x → (batch_size, channels, frames, height, width)

        # Extract spatiotemporal features
        features = self.backbone(x)

        # Apply dropout
        features = self.dropout(features)

        # Map features to class scores
        output = self.fc(features)

        return output
