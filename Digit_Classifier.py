import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
# This is your neural network class, so it must extend nn.Module
# For your final submission you will be submitting this cell as its own file
# REMEMBER TO REPLACE X IN THE CLASS NAME WITH YOUR TEAM NUMBER
class Digit_Classifiertbd(nn.Module):
  def __init__(self):
    # Handle some under-the-hood PyTorch stuff
    super().__init__()
    # Now put your layers below in addition to any other member variables you need
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 24, 5)
    self.conv3 = nn.Conv2d(16, 30, 5)
    self.flat = nn.Flatten()
    self.layer1 = nn.Linear(24 * 4 * 4, 100)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(100, 80)
    self.layer3 = nn.Linear(80, 10)

  def forward(self, x):
    # Now here you add your forward pass, e.g. how the layers fit together
    # Tips:
    # 1. Don't forget the ReLU layer when needed
    # 2. Consider normalization
    # 3. If you are getting errors look at your dimensions, dimension errors are very easy to make!
    # 4. CNN layers take in rectangular (or square) inputs and give rectangular (or square) outputs. Fully connected layers have input and output that are vectors, when you need to switch between the two consider using a flatten or reshape
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.relu(self.conv2(x))
    x = self.pool(x)
    x = self.flat(x)
    x = self.relu(self.layer1(x))
    x = self.relu(self.layer2(x))
    x = self.layer3(x)
    return x
  
  # Optional: any other member functions that you think would be helpful

class Model:
    def __init__(self):
       self.model = Digit_Classifiertbd()
       self.model.load_state_dict(torch.load('./Weights/teamtbd_final_weights.pth'))
       self.model.eval()
       self.convert_tensor = torchvision.transforms.ToTensor()

    def predict(self, img):
       return torch.argmax(self.model(torch.reshape(self.convert_tensor(img), (1, 1, 28, 28))))

# img = Image.open("./test.png")
# img = ImageOps.invert(img)
# convert_tensor = torchvision.transforms.ToTensor()
# img = convert_tensor(img)
# img = torch.reshape(img, (1, 28, 28))

# plt.imshow(img.numpy().squeeze(), cmap="gray_r")
# plt.show()