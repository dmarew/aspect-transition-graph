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
import glob

def main():
    encoder_time = time.time()
    number_of_epochs = 30
    batch_size = 32
    dataset_path = 'data/real_aspects/*'
    number_of_samples = 500
    autoencoder = nn.Sequential(Encoder(), Decoder())
    dataset = glob.glob(dataset_path)
    train_ds = ATGDataset(dataset[:500])
    train_loader = data.DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    test_ds = ATGDataset(dataset[500:])
    test_loader = data.DataLoader(test_ds, batch_size = 1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    test_recon_over_time = []
    training_loss_history = []
    for epoch in range(number_of_epochs):

        running_loss = 0.0

        for batch_index, in_images in enumerate(train_loader):

            in_images = to_var(in_images)
            out_images = autoencoder(in_images)
            loss = criterion(out_images, in_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy()
            if batch_index % 10:
                print('epoch %d loss: %.5f batch: %d' % (epoch, running_loss/((batch_index + 1)*batch_size), (batch_index + 1)*batch_size))
                training_loss_history.append(running_loss/((batch_index + 1)*batch_size))


        dataiter = iter(test_loader)
        in_image = dataiter.next()
        decoded_img = autoencoder(to_var(in_image))

        test_recon_over_time.append(decoded_img)


    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")
    autoencoder = nn.Sequential(Encoder(), Decoder())
    autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    dataiter = iter(test_loader)
    in_images = dataiter.next()
    decoded_imgs = autoencoder(to_var(in_images))

    epoch_count = range(1, len(training_loss_history) + 1)
    plt.plot(epoch_count, training_loss_history, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.show()

    print('training  encoder took %.2f secs'%(time.time()-encoder_time))
    test_recon_over_time.append(in_images)
    test_recon_over_time = torch.stack(test_recon_over_time).squeeze(1)
    imshow(torchvision.utils.make_grid(test_recon_over_time.data), True)

if __name__ =='__main__':
    main()
