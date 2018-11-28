from skimage import io, transform
import torch
from torchvision import transforms
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, afflictions = sample['image'], sample['afflictions']

        img = transform.resize(image, (self.output_size,self.output_size))

        #print ("Rescale: ###########",img.shape)
        return {'image': img, 'afflictions': afflictions}

class Normalize(object):


    def __init__(self, parameter1, parameter2):
        self.parameters1 = parameter1
        self.parameters2 = parameter2

    def __call__(self, sample):
        image, afflictions = sample['image'], sample['afflictions']
        image = torch.from_numpy(image)
        function = transforms.Normalize(self.parameters1,self.parameters2)
        img = function(image).permute(2,0,1).float()
        # img = img.permute(2,0,1).float()
        #img = img.numpy()
        #print ("Normalize: ###########",img.shape)
        #print ("###########",img.shape)
        return {'image': img, 'afflictions': afflictions}
