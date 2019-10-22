from utils import *
import torch
import torchvision.transforms
from models import *

if __name__=='__main__':
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    encoder = autoencoder[0]
    print('encoder', encoder)
    get_encoder_feature_for_dataset(encoder,
                                        num_workers=6,
                                        dataset_path = 'data/atg_dataset.npz',
                                        output_path='data/atg_dataset_encoder_feat.npz')
