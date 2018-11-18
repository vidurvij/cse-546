import pandas as pd
from skimage import io
from tqdm import tqdm
csv = pd.read_csv("Data_Entry_2017.csv")
l = len(csv)
csv2 = pd.DataFrame()
for i in tqdm(range(l)):
    name = "/home/vidur/Desktop/images/" + csv.iloc[i,0]
    image = io.imread(name)
    if image.shape != (1024,1024):
        continue
    csv2 = csv2.append(csv.iloc[i])
    #print(csv2)

csv2.to_csv("Data_Entry.csv")
