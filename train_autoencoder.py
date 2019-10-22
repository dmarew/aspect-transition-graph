import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models import *
from utils import *
from data_loader import *

def main():
    number_of_epochs = 1
    batch_size = 32

    autoencoder = nn.Sequential(Encoder(), Decoder())
    dataset = np.load('./data/atg_dataset.npz')
    num_samples = dataset['data'].shape[0]
    ds = ATGDataset(num_samples, dataset=dataset)
    test_loader = data.DataLoader(ds, batch_size = batch_size, shuffle = True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())


    for epoch in range(number_of_epochs):

        running_loss = 0.0

        for batch_index, (in_images, _, _) in enumerate(test_loader):

            in_images = to_var(in_images)
            out_images = autoencoder(in_images)
            loss = criterion(out_images, in_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if batch_index % 100:
                print('avg loss: ', running_loss/((batch_index + 1)*batch_size), (batch_index + 1)*batch_size)
                pass
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    dataiter = iter(test_loader)
    (in_images, _, _) = dataiter.next()
    decoded_imgs = autoencoder(to_var(in_images))
    imshow(torchvision.utils.make_grid(decoded_imgs.data), True)
if __name__ =='__main__':
    main()
