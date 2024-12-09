import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, model='vgg16', layers=None):
        super(PerceptualLoss, self).__init__()
        if model == 'vgg16':
            vgg = models.vgg16(pretrained=True).features
        elif model == 'vgg19':
            vgg = models.vgg19(pretrained=True).features
        else:
            raise ValueError("Unsupported model. Use 'vgg16' or 'vgg19'.")

        if layers is None:
            self.layers = [0, 5, 10, 19, 28]  # Default layers for perceptual loss
        else:
            self.layers = layers

        # Extract only the required layers in sequence
        self.model = nn.Sequential(*[vgg[i] for i in range(max(self.layers) + 1)]).to("cuda")
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze the perceptual model

        self.layer_mapping = {layer: idx for idx, layer in enumerate(self.layers)}

    def preprocess(self, x):
        # Resize to (224, 224) and normalize for VGG
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)) / \
            torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return x

    def forward(self, x, y):
        x, y = self.preprocess(x), self.preprocess(y)
        
        loss = 0
        outputs_x, outputs_y = [], []
        
        # Propagate through the sequential model
        for i, layer in enumerate(self.model):
            x, y = layer(x), layer(y)
            if i in self.layers:
                outputs_x.append(x)
                outputs_y.append(y)
        
        # Compute perceptual loss for selected layers
        for feat_x, feat_y in zip(outputs_x, outputs_y):
            loss += nn.functional.mse_loss(feat_x, feat_y)
        
        return loss
