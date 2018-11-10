from torch.utils.data import Dataset
import pandas as pd
import os
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
                                self.afflictions_root.iloc[idx, 0])
        #print("11111111111111111111111111111111111:",img_name)
        image = io.imread(img_name)
        image = image.reshape(1,image.shape[0],image.shape[1])
        afflictions = self.afflictions_root.iloc[idx, 1]
        afflictions = afflictions.split('|')
        index = [self.directory.index(x) for x in afflictions]
        afflictions = np.zeros(len(self.directory))
        afflictions[index] = 1
        sample = {'image': image, 'afflictions': afflictions}

        if self.transform:
            sample = self.transform(sample)
        return sample


    #sdef view(index):
