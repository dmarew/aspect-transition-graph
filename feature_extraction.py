from utils import *
import torch
import torchvision.transforms
from models import *

if __name__=='__main__':
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    encoder = autoencoder[0]
    print('encoder', encoder)
    #get_vgg_feature_for_dataset(vgg16, num_workers=6, dataset_path = './data/atg_dataset.npz')
    get_encoder_feature_for_dataset(encoder, num_workers=6, dataset_path = './data/atg_dataset.npz')
