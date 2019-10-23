import os
import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from models import *
from torchvision import transforms
from PIL import Image
from utils import *
from skimage.transform import warp
from skimage.transform import SimilarityTransform

class ATGDataset(data.Dataset):
    """ ATG dataset """

    def __init__(self, dataset, image_size=64):

        super(ATGDataset, self).__init__()

        self.dataset   = dataset
        self.transformer = transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor()])

    def __getitem__(self, index):
        """
        returns
        """
        image = Image.open(self.dataset[index])
        image = self.transformer(image)
        return image
    def __len__(self):
        """length of dataset"""
        return len(self.dataset)

class AspectNodeDataset(data.Dataset):
    """ Aspect node dataset """

    def __init__(self, dataset, labels):

        super(AspectNodeDataset, self).__init__()
        self.data = dataset['data']
        self.labels = labels

    def __getitem__(self, index):
        """
        returns
        """
        if index == 0:
            input_feat = torch.from_numpy(np.array([self.labels[index], self.data[index][2]])).float()
            return input_feat, self.labels[index]
        input_feat = torch.from_numpy(np.array([self.labels[index], self.data[index][2]])).float()
        return input_feat, self.labels[index]

    def __len__(self):
        """length of dataset"""
        return self.data.shape[0]

if __name__ == '__main__':
    dataset = np.load('./data/atg_dataset.npz')
    actions = dataset['data'][:, 2]

    labels = np.load('data/encoder_clustering_labels.npz')['clustering_labels'][1:]
    atg_dataset_encoder_feat = np.load("data/atg_dataset_encoder_feat.npz")['deep_feat']
    num_samples = dataset['data'].shape[0]
    ds = ATGDataset(num_samples, dataset=dataset, actions = actions, features = atg_dataset_encoder_feat, labels=labels)
    test_loader = data.DataLoader(ds, batch_size = 4, shuffle = True)

    for batch_index, (images, feats, labels) in enumerate(test_loader):
        print(batch_index, images.shape, feats.shape, labels)
        break
