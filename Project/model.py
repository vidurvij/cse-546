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


def train_model(model,train, valid, criterion, optimizer, scheduler, num_epochs=25):
    batch_size = train.batch_size
    writer = SummaryWriter()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    model.train()

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
            outputs1 = torch.round(torch.sigmoid(outputs))
            # print(outputs)
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
            running_corrects += torch.equal(labels[0:batch_size],outputs[0:batch_size])
            #print (running_corrects)
        for i, sample in enumerate(valid) :
            inputs, labels = sample['image'], sample['afflictions']
            outputs = model(inputs)
            outputs = torch.round(torch.sigmoid(outputs))
            #print(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.equal(labels[0:batch_size],outputs[0:batch_size])
            # print(running_corrects)


        epoch_loss = running_loss / len(train)
        epoch_acc = running_corrects / len(train)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, epoch_loss, epoch_acc))
        writer.add_scalar('data/scalar1', epoch_loss, i)
        writer.add_scalar('data/scalar2', epoch_acc, i)
        if  epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
