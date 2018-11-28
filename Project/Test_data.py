import numpy as np
import torch as tp
from sklearn.metrics import roc_curve, auc
from XrayDataset import XrayDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from Transforms import Rescale, Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = XrayDataset(csv_file = "Data_Entry.csv",root_dir = "/home/vidur/Desktop/images",transform = transforms.Compose([Rescale(224),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
Test_loader = DataLoader(dataset, batch_size = 50)

# model = tp.load("model.pth")
# model.eval()
# outputs = []
# for i, sample in enumerate(tqdm(Test_loader)):
#     inputs, labels = sample['image'], sample['afflictions']
#     output = tp.round(tp.sigmoid(model(inputs)))
#     output =  (output.detach().numpy()).astype(int)
#     print(output.shape)
#     # outputs.append(output)
# np.save("outputs.npy",np.array(outputs))

lab = dataset.get_labels()
fig, ax = plt.subplots(1,1, figsize = (9,9))
res = np.load("outputs.npy")
a = res[0]
for i in range(1,len(res)):
    a = np.append(a,res[i], 0)

res = a

for idx, label in enumerate(dataset.directory):
    fpr, tpr, thresholds = roc_curve(lab[:,idx],res[:,idx])
    ax.plot(fpr, tpr, label = label+" (AUC:" + str(auc(fpr,tpr))+")")
ax.legend()
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
fig.savefig("Dekhte Hain")
