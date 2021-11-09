from torchvision.models import vgg19
import torch
import torch.nn as nn

class VGG19Feature(nn.Module):
    layer_name = [
        'conv1_1', 'relu1_1', 
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'conv3_4', 'relu3_4',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'conv4_4', 'relu4_4',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'conv5_4', 'relu5_4',
        'pool5',
    ]

    def __init__(self, layer='relu5_4'):
        super(VGG19Feature, self).__init__()
        index = self.layer_name.index(layer)
        self.net = vgg19(pretrained=True).features[:index + 1]
        self.net.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)