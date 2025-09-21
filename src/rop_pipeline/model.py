import torch
import torch.nn as nn
import timm


class EffNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier with custom head.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for the head. Default: 0.3
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Create pretrained EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0", 
            pretrained=True, 
            num_classes=0,  # Remove the original classifier
            global_pool="avg"  # Global average pooling
        )
        
        # Get the number of features from the backbone
        in_features = self.backbone.num_features
        
        # Create custom head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Logits of shape [B, num_classes]
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply custom head
        logits = self.head(features)
        
        return logits


def freeze_backbone(model: EffNetClassifier, freeze: bool = True) -> None:
    """
    Toggle requires_grad for backbone parameters only.
    
    Args:
        model (EffNetClassifier): The model to modify
        freeze (bool): If True, freeze backbone parameters. If False, unfreeze them.
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    # Ensure head parameters are always trainable
    for param in model.head.parameters():
        param.requires_grad = True


# Export both classes/functions
__all__ = ["EffNetClassifier", "freeze_backbone"]
