import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool, **kwargs):
        super().__init__()

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        return self.model(x)


    def freeze_layers(self, freeze_option):
        unfreeze_layers = []
        if freeze_option == "unfreeze_penultimate":
            unfreeze_layers.append("layer4")
        elif freeze_option == "unfreeze_last":
            unfreeze_layers.append("fc")
        elif freeze_option == "unfreeze_both":
            unfreeze_layers.extend(["layer4", "fc"])

        for name, param in self.model.named_parameters():
            param.requires_grad = False 

        for name, param in self.model.named_parameters():
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True 

