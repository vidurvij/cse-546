import pandas as pd
from skimage import io
from tqdm import tqdm

csv = pd.read_csv("Data_Entry.csv")
l = len(csv)
csv2 = pd.DataFrame()
for i in tqdm(range(l)):
    name = "/home/vidur/Desktop/images/" + csv.iloc[i,4]

    image = io.imread(name)
    assert image.shape == (1024,1024)
