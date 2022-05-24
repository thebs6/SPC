import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PDataSet(Dataset):
    def __init__(self, args, csv_path, split, transform):
        super(PDataSet, self).__init__()
        self.args = args
        self.df = pd.read_csv(csv_path, dtype={'imageId': str})
        self.df['image_path'] = self.df['imageId'].apply(lambda p: os.path.join(args.image_folder, p + '.jpg'))
        self.transform = transform
        if split not in ['train', 'valid', 'test']:
            raise (ValueError, 'split error')
        self.resdf = self.df[self.df['split'] == split]
        self.classes = self.resdf['plane_type'].unique()

    def __len__(self):
        return len(self.resdf)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.args.image_folder, self.resdf.iloc[item]['imageId'] + '.jpg'))
        image = self.transform(image)
        label = self.resdf.iloc[item]['label']
        return image, label

