class XrayDataLoader(Dataset):

    def __init__(self, csv_file, image_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.affliction = pd.read_csv(csv_file)
        self.image_file = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.affliction)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.affliction.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
