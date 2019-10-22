import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
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
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
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

class AspectTransitionModel(nn.Module):

    def __init__(self, number_of_aspect_nodes):
        """Aspect Transition Model"""
        super(AspectTransitionModel, self).__init__()
        self.layers = nn.ModuleList([
		nn.Linear(6920, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, number_of_aspect_nodes)
        ])

    def forward(self, input):
        """Extract the image feature vectors."""
        features = input
        for layer in self.layers:
            features = layer(features)
        return features

class AspectNodeTransitionModel(nn.Module):

    def __init__(self, number_of_aspect_nodes):
        """Aspect Transition Model"""
        super(AspectNodeTransitionModel, self).__init__()
        self.layers = nn.ModuleList([
		nn.Linear(2, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, number_of_aspect_nodes)
        ])

    def forward(self, input):
        """Extract the image feature vectors."""
        features = input
        for layer in self.layers:
            features = layer(features)
        return features

if __name__ =='__main__':
    encoder = Encoder()
    decoder = Decoder()
    X = Variable(torch.randn(4, 3, 144, 192))
    encoded = encoder(X)
    print(X.shape, encoded.shape, decoder(encoded).shape)
    at_model = AspectTransitionModel(72)
    X = Variable(torch.randn(4, 6920))
    print(X.shape, at_model(X).shape)
