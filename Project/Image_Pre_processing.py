import torch as tp
import pandas as pd
from XrayDataset import XrayDataset
from torchvision import transforms
from Transforms import Rescale, Normalize
from tqdm import tqdm
# from skimage import io

dataset = XrayDataset(csv_file = "Data_Entry.csv",root_dir = "/home/vidur/Desktop/images",transform = transforms.Compose([Rescale(224),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
# a = dataset[0]
# pd.Series(a['image'])
# main = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in a.items()]))
# main = pd.DataFrame(dataset[0])
print(dataset)

for i in tqdm(range(len(dataset))):
    tp.save(dataset[i]['image'],"/home/vidur/Desktop/ImageXrayRescaled/"+ dataset[i]['name'][:-3]+"pt")
