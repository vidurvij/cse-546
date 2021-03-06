from DataLoad import *
from Neural_Net import *
import numpy as np
from utils import *
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm


# A part
rate = .005
momentum = 0.5
M = 200
N = 6
net_a = Net(4, M = M, p = 5, p2 = 4, N = N)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_a.parameters(), lr=rate, momentum=momentum)
epochs = 50
<<<<<<< HEAD
runner = 0
writer = SummaryWriter("runse/N4Base--Loss")
writer2 = SummaryWriter("runse/N4Base--Loss")
=======
#runner = 0
writer = SummaryWriter("runsd/N4Base-15Loss")
writer2 = SummaryWriter("runsd/N4Base-15-Loss")
>>>>>>> 665b3f83c5fc7ea560a015a2564cbb762c7e517e
for step in tqdm(range(epochs)):
    running_loss = 0.0
    train_acc = 0
    for i, data in enumerate(tqdm(trainloader)):
        image, label = data
        #Pre Process
        # label = onehot(label,classes)
        # image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
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

        #runner += 1
        predictions = torch.argmax(output,1)
        train_acc += torch.sum(torch.eq(predictions,label)).item()
    writer.add_scalar('loss', running_loss, step)
    train_acc /= len(trainset)
    # print(train_acc)
    test_acc = 0

    for i, data in enumerate(testloader):
        image, label = data
        #Pre Process
        # label = onehot(label,classes)
        # image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
        output = net_a(image)
        predictions = torch.argmax(output,1)
        test_acc += torch.sum(torch.eq(predictions,label)).item()
    test_acc /= len(testset)
    writer.add_scalar('Training_accuracy', train_acc, step)
    writer.add_scalar('Validation_accuracy', test_acc, step)
