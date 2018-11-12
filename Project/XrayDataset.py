from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io, transform
import numpy as np
class XrayDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.afflictions_root = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.directory = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation','No Finding']

    def __len__(self):
        return len(self.afflictions_root)

    def __getitem__(self, idx):
        idx = int(idx)
        img_name = os.path.join(self.root_dir,
                                self.afflictions_root.iloc[idx, 4])
        #print("11111111111111111111111111111111111:",img_name)
        image = io.imread(img_name)
        image = image.reshape(image.shape[0],image.shape[1],1)
        image = np.repeat(image,3,-1)
        afflictions = self.afflictions_root.iloc[idx, 1]
        afflictions = afflictions.split('|')
        index = [self.directory.index(x) for x in afflictions]
        afflictions = np.repeat(-1,len(self.directory))
        afflictions[index] = 1
        afflictions = torch.from_numpy(afflictions).float()
        sample = {'image': image, 'afflictions': afflictions}
        #print ("Original: ###########",image.shape)
        if self.transform:
            sample = self.transform(sample)
        return sample


    #sdef view(index):
