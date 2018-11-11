from XrayDataset import XrayDataset
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from Transforms import Rescale, Normalize
from torchvision import transforms, utils, models
import matplotlib.pyplot as plt
from model import *

import warnings
warnings.filterwarnings("ignore")

#dataset = XrayDataset(csv_file = "Data_Entry_2017.csv",root_dir = "/home/vidur/Desktop/images")
dataset = XrayDataset(csv_file = "Data_Entry.csv",root_dir = "/home/vidur/Desktop/images",transform = transforms.Compose([Rescale(224),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
train, test = random_split(dataset,lengths = (int(.8*len(dataset)),len(dataset)-int(.8*len(dataset))))
train, valid = random_split(train,lengths = (int(.9*len(train)),len(train)-int(.9*len(train))))
print(valid[1]['image'].shape)
model_ft = models.resnet152(pretrained=True)

# for i,parameter in enumerate(valid):
#      print(i)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 15)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(model_ft.fc.parameters(), .9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
Train_loader = DataLoader(train,batch_size = 5)
Valid_loader = DataLoader(valid,batch_size = 5)

model_ft = train_model(model_ft,Train_loader,Valid_loader,criterion,optimizer,exp_lr_scheduler)
