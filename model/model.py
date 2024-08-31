import torch
import torch.nn as nn
from torchvision import models

class AdvancedEnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AdvancedEnsembleModel, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        densenet_out = self.densenet(x)
        efficientnet_out = self.efficientnet(x)
        combined_out = torch.cat((densenet_out, efficientnet_out), dim=1)
        final_out = self.meta_learner(combined_out)
        return final_out
