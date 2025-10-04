import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data import train_loader, test_loader

# choose GPU > CPU if availbile & set seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

if device.type == "cuda":
    torch.cuda.manual_seed(42)

# Collect & store labels in train, test
labels_list=[]
for _, labels in train_loader:
    labels_list.append(labels)
for _, labels in test_loader:
    labels_list.append(labels)

labels_list = torch.cat(labels_list)    # concatenate all label tensors into 1D tensor
num_classes = int(torch.unique(labels_list).numel())
print(f"Detected number of classes: {num_classes}")


'''
Conv2D -> creates grids of number, higher values = stronger activations (detection of edges/texture, etc.)
    - in_channels = number of channels in the input image (1 - Greyscale, 3 - RGB)
    - out_channels = number of filters (feature maps) the layer learns
    - kernel_size = size of the conv filter (n x n) or tuple: (h x w)
    - padding = adds pixels around the input

MaxPool2d -> downsamples feature maps by sliding kernel (n x n) over them and for each window replacing w/ single maximum value
    - kernel_size = size of the window over what the max is taken

EXAMPLE:

Feature Map:
    1  3  2  4
    5  6  1  2
    7  2  8  3
    4  9  2  1

MaxPool2d(2) -> 2x2 kernels 
   
    | 1  3 | 2  4
    | 5  6 | 1  2
      7  2   8  3
      4  9   2  1

RESULTS:
    6  4
    9  8
      

'''
class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), #
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def foward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(train_loader, test_loader, in_channels, num_classes, epochs=5, device=device):
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader): .4f}")


    model.eval()
    correct, total = 0,0
