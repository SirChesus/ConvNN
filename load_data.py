import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Dataset

import os
from dotenv import load_dotenv

load_dotenv()


Images = np.load(os.getenv("IMAGES"))
Labels = pd.read_csv(os.getenv("LABELS")).values.squeeze()

'''
print(f"Image dataset shape: {Images.shape}")   # ( Number of samples, Height, Width, Colour (1 | 3) )
print(f"Label dataset shape: {Labels.shape}")  # (Number of samples)
'''

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, index):
        img = torch.tensor(self.images[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # Converts any (H, W, C) to (C, H, W) 
        if img.ndim == 3 and img.shape[-1] in (1,3):
            img = img.permute(2,0,1)

        return img, label


dataset = ImageDataset(Images, Labels)

total_size = len(dataset)
train_size = int(0.8*total_size)
test_size = total_size - train_size

train, test = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

if __name__ == "__main__":
    imgs, labels = next(iter(train_loader))
    
    print(f"Batch images shape {imgs.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Example labels: {labels[:10]}")