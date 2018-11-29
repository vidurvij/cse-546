import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,part, M = 1, p = 1, N = 1):
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
            self.pool = nn.MaxPool2d(N)
            self.fc1 = nn.Linear(int(M*((33-p)/N)**2),10, True)
            self.execute = self.c

        if part == 4:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.fc1(x)
        return x

    def d(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
