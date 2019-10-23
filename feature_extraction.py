from utils import *
import torch
import torchvision.transforms
from models import *

if __name__=='__main__':
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.eval()
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    encoder = autoencoder[0]
    print('encoder', encoder)
    get_encoder_feature_for_dataset(encoder,
                                        num_workers=1,
                                        dataset_path = 'data/real_aspects/',
                                        output_path='data/real_dataset_encoder_feat.npz')
