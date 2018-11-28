import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
from torchviz import make_dot


def train_model(model,train, valid, criterion, optimizer, num_epochs=3):
    np.set_printoptions(precision = 4)
    batch_size = train.batch_size
    writer = SummaryWriter()
    since = time.time()
    len(train)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = torch.zeros(15)
    model.train()
    total = len(train.dataset)+len(valid.dataset)
    print(total)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, sample in enumerate(train):
            # print(i)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            inputs, labels = sample['image'], sample['afflictions']
            #print(labels)
            # zero the parameter gradients
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            outputs = model(inputs)

            # print(outputs)
            outputs1 = torch.sigmoid(outputs)
            # outputs1 = torch.round(outputs1)
            loss = criterion(outputs, labels)

            # make_dot(loss).view()
            # exit()
            #backward + optimize only if in training phase
            loss.backward()
            print(loss)
            #print(loss.backward())
            # for key,value in model.state_dict().items():
            #     print(value.grad)
            optimizer.step()

            # statistics
            # print(labels ==  sum)
            running_loss += loss.item() * inputs.size(0)
            # print(type(running_corrects))
            # print(torch.round(outputs1))
            running_corrects += torch.sum(torch.eq(labels,torch.round(outputs1)),0)
            print
            # print(list(running_corrects.numpy()))
            # print (running_corrects.type())
        for i, sample in enumerate(valid) :
            inputs, labels = sample['image'], sample['afflictions']
            outputs2 = model(inputs)
            outputs3 = torch.sigmoid(outputs2)
            #print(outputs)
            # outputs3 = torch.round(outputs3)
            loss1 = criterion(outputs2, labels)
            running_loss += loss1.item() * inputs.size(0)
            # print(running_corrects)
            running_corrects += torch.sum(torch.eq(labels,torch.round(outputs3)),0)
            # print(running_corrects)


        # print(running_corrects)
        epoch_loss = torch.div(torch.tensor(running_loss),(total))
        epoch_acc = torch.div(running_corrects.float(),(total))
        # print(epoch_acc.type())
        # print(epoch_acc.type(),epoch_loss.type())
        print('{} Loss: {:.4f} Acc:'.format(
            epoch, epoch_loss))

        print(epoch_acc)
        # writer.add_scalar('data/scalar1', epoch_loss, i)
        # writer.add_scalar('data/scalar2', epoch_acc, i)
        if  torch.mean(epoch_acc.float()) > torch.mean(best_acc.float()):
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4s}'.format(str(best_acc.numpy())))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
