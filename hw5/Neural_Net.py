import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self,part, M = 1, p = 1, N = 7, p2 = 5):
        if part == 1 :
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3072, 10, True)
            self.execute = self.a
        if part == 2:
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3072, M, True)
            self.fc2 = nn.Linear(M, 10, True)
            self.execute = self.b

        if part == 3:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, M, p, bias = True)
            self.pool = nn.MaxPool2d(p2,N)
            self.fc1 = nn.Linear(int(M*((((28-p2)/N)+1)**2)),10, True)
            self.execute = self.c

        if part == 4:
            super(Net, self).__init__()
<<<<<<< HEAD
            self.conv1 = nn.Conv2d(3,96,3)
            self.conv2 = nn.Conv2d(96,96,3)
            self.pool  = nn.MaxPool2d(3,2)
            self.conv3 = nn.Conv2d(96,192,3)
            self.conv4 = nn.Conv2d(192,192,3)
            self.conv5 = nn.Conv2d(192,192,1)
            self.conv6 = nn.Conv2d(192,10,1)
            self.gpool = nn.AvgPool2d(2)
=======
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.conv3 = nn.Conv2d(32,256,5)
            self.fc1 = nn.Linear(256 * 1 * 1, 120)
            #self.batch1 = nn.BatchNorm1d(32*5*5)
            self.fc2 = nn.Linear(120, 84)
            #self.batch2 = nn.BatchNorm1d(120)
            self.fc3 = nn.Linear(84, 10)
            #self.batch3 = nn.BatchNorm1d(84)
            #self.soft = nn.Softmax(0)
>>>>>>> 665b3f83c5fc7ea560a015a2564cbb762c7e517e
            self.execute = self.d

    def forward(self, x):
        x = self.execute(x)
        return x

    def a(self,x):
        x = self.fc1(x)
        return x

    def b(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def c(self,x):
        # print("@@@@@@@@@@@",(self.conv1(x)).shape)
        # exit()
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        # print(x.shape)
        # exit()
        x = self.fc1(x)
        return x

    def d(self,x):
<<<<<<< HEAD
        x = (torch.relu(self.conv1(x)))
        x = (torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = (torch.relu(self.conv3(x)))
        # print(x.shape)
        x = (torch.relu(self.conv4(x)))
        x = self.pool(x)
        x = (torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.gpool(x)
        # print(x.shape)
        # exit()
        # x = x.view(-1, 32 * 10 * 10)
        # x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        # x = self.fc3(x)
        x = x.reshape(x.shape[0],10)

=======
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print("............",x.shape)
        #exit()
        x = x.view(-1, 256 * 1 * 1 )
        #x = self.batch1(x)
        x = torch.sigmoid(self.fc1(x))
        #x = self.batch2(x)
        x = torch.sigmoid(self.fc2(x))
        #x = self.batch3(x)
        x = (self.fc3(x))
>>>>>>> 665b3f83c5fc7ea560a015a2564cbb762c7e517e
        return x
