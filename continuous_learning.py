import sys
import os
import time
import argparse

from train_autoencoder import *
from utils import *
from feature_extraction import *
from clustering import *

def main(args):
    dataset_path = 'data/sample/'

    while True:
        number_of_images = len(os.listdir(dataset_path))
        print('So far %d observations have been collected'%(number_of_images))
        if number_of_images > 0 and number_of_images % 600 == 0:
            print('Training autoencoder %d images ...'%(number_of_images))
            train_auto_encoder(number_of_epochs=10,
                               batch_size = 8,
                               dataset_path = dataset_path,
                               number_of_samples = number_of_images,
                               image_size = 256,
                               verbose=True)
            print('Done training autoencoder')
        else:
            print('Waiting till I get enough samples for retraining')


if __name__ == '__main__':
    main(sys.argv[1:])
