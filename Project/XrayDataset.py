from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io, transform

class XrayDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.afflictions_root = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.afflictions_root)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.afflictions_root.iloc[idx, 0])
        #print("11111111111111111111111111111111111:",img_name)
        image = io.imread(img_name)
        afflictions = self.afflictions_root.iloc[idx, 1].as_matrix()
        print("::::::::::",afflictions)
        afflictions = afflictions.astype('float').reshape(-1, 2)
        #print("::::::::::------------------",afflictions.shape)
        sample = {'image': image, 'afflictions': afflictions}

        if self.transform:
            sample = self.transform(sample)
        return sample


    #sdef view(index):
