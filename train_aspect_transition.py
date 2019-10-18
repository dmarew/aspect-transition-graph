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
    number_of_epochs = 5
    batch_size = 128


    dataset = np.load('./data/atg_dataset.npz')
    actions = dataset['data'][:, 2]
    labels = np.load('data/encoder_clustering_labels.npz')['clustering_labels']
    atg_dataset_encoder_feat = np.load("data/atg_dataset_encoder_feat.npz")['deep_feat']
    num_samples = labels.shape[0]
    number_of_aspect_nodes = np.max(labels) + 1

    at_model = AspectTransitionModel(number_of_aspect_nodes)

    ds = ATGDataset(num_samples, actions=actions, features = atg_dataset_encoder_feat, labels=labels)

    test_loader = data.DataLoader(ds, batch_size = batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(at_model.parameters(), lr = 1e-1)
    training_loss_history = []
    training_acc_history = []


    for epoch in range(number_of_epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for batch_index, (_, features, labels) in enumerate(test_loader):
            features, labels = to_var(features), to_var(labels)
            optimizer.zero_grad()
            transition_prob = at_model(features)
            _, predicted = transition_prob.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss = criterion(transition_prob, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if batch_index % 10==0:
                training_loss_history.append(running_loss/((batch_index + 1)*batch_size))
                training_acc_history.append(correct/total)
                #print('training loss: ', running_loss/((batch_index + 1)*batch_size), (batch_index + 1)*batch_size)
                #print('training acc : ', 100.*correct/total)

    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    torch.save(at_model.state_dict(), "./weights/at_model.pkl")
    at_model = AspectTransitionModel(number_of_aspect_nodes)
    at_model.load_state_dict(torch.load("./weights/at_model.pkl"))
    epoch_count = range(1, len(training_loss_history) + 1)
    plt.plot(epoch_count, training_loss_history, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(epoch_count, training_acc_history, 'b-')
    plt.legend(['Training Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ =='__main__':
    main()
