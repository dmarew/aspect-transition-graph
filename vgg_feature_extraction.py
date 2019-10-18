from utils import *
import torch
import torchvision.transforms

if __name__=='__main__':
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    get_vgg_feature_for_dataset(vgg16, num_workers=6, dataset_path = './data/atg_dataset.npz')
