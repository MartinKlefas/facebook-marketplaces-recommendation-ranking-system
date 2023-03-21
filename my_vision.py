import os, json
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2

class classification_dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        with open("image_decoder.json","r") as f:
            self.lookup = json.load(f)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = cv2.imread(img_path + ".jpg")
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label    
    
    def human_category_label(self,category : str):
        category = str(category)
        return self.lookup[category]
    
    def machine_category_label(self,category : str):
        category = str(category)
        return self.lookup[category]