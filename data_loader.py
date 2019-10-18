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

class ATGDataset(data.Dataset):
    """ ATG dataset """

    def __init__(self, num_samples, dataset=None, actions = None, features=None, labels=None):

        super(ATGDataset, self).__init__()

        self.dataset   = dataset
        self.labels = labels
        self.features = features
        self.num_samples = num_samples
        self.actions = actions

    def __getitem__(self, index):
        """
        returns
        """
        image = 0
        label = 0
        feature = 0

        if self.dataset is not None:
                images_path = str(self.dataset['images_path'])
                image = np.array(imageio.imread(images_path + str(self.dataset['data'][index][3]) + '.png'))/255.0
                image = torch.from_numpy(image).transpose(2,1).transpose(1,0).float()
        if self.labels is not None:
            label = self.labels[index]
        if self.features is not None:

            if index > 0:
                feature = self.features[index-1]
            else:
                feature = self.features[index]
            action_one_hot = np.zeros(np.max(self.actions)+1)
            action_one_hot[self.actions[index]] = 1.
            feature = torch.from_numpy(np.hstack((feature, action_one_hot))).float()

        return image, feature, label

    def __len__(self):
        """length of dataset"""
        return self.num_samples

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
