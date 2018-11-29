from DataLoad import *
from Neural_Net import *
import numpy as np
from utils import *
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm


# A part
rate = .0015
momentum = 0
net_a = Net(1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_a.parameters(), lr=rate, momentum=momentum)
epochs = 50
runner = 0
writer = SummaryWriter("runs/N1l"+str(rate)+"m"+str(momentum)+"-Loss")
writer2 = SummaryWriter("runs/N1l"+str(rate)+"m"+str(momentum)+"-Accuracy")
for step in tqdm(range(epochs)):
    running_loss = 0.0
    train_acc = 0
    for i, data in enumerate(trainloader):
        image, label = data
        #Pre Process
        # label = onehot(label,classes)
        image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
        optimizer.zero_grad()
        #Foward Pass
        output = net_a(image)
        # print(output.shape,label.shape)
        # print(label)
        # exit()
        #Backward Pass
        loss = criterion(output,label.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        writer.add_scalar('loss', running_loss, runner)
        runner += 1
        predictions = torch.argmax(output,1)
        train_acc += torch.sum(torch.eq(predictions,label)).item()
    train_acc /= len(trainset)
    # print(train_acc)
    test_acc = 0

    for i, data in enumerate(testloader):
        image, label = data
        #Pre Process
        # label = onehot(label,classes)
        image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
        output = net_a(image)
        predictions = torch.argmax(output,1)
        test_acc += torch.sum(torch.eq(predictions,label)).item()
    test_acc /= len(testset)
    writer.add_scalar('Training_accuracy', train_acc, step)
    writer.add_scalar('Validation_accuracy', test_acc, step)
