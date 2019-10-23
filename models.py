import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self):
        """Encoder"""
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
        nn.Conv2d(3, 32, kernel_size = 3, stride=2, padding=1),
        nn.ReLU(),
        torch.nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size = 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride=2, padding=1),
        nn.ReLU()
        ])

    def forward(self, images):
        """Extract the image feature vectors."""
        features = images
        for layer in self.layers:
            features = layer(features)
        return features

class Decoder(nn.Module):

    def __init__(self):
        """Decoder"""
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
		nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
		nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
		nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
		nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        nn.Sigmoid()
        ])

    def forward(self, images):
        """Extract the image feature vectors."""
        features = images
        for layer in self.layers:
            features = layer(features)
        return features
