import os

import torch
from torchvision import datasets, transforms, models
import numpy as np
from torch import nn
import torch.nn.functional as F
import requests
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

classes=('cat','dog')

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1,padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1,padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1,padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1,padding=1)
        # self.conv5 = nn.Conv2d(128, 256, 3, 1,padding=1)
        self.fc1 = nn.Linear(8 * 8 * 128, 500)
        self.dropout1=nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv5(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8 * 8 * 128)
        x=self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def modelInitialize():
    global model
    model = nn()
    pth=os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(pth+'/cnn_cat_dog.pth'))
    return model

def guessTheImage(image):
    # model=modelInitialize()
    # model = LeNet().to(device)
    pth = os.path.dirname(os.path.realpath(__file__))
    checkpoint=torch.load(pth + '/cnn_cat_dog.pth')

    print('checkpoint is ---------',checkpoint)
    model=checkpoint['state_dict']
    img = Image.open(image)
    img = transform(img)
    image = img.to(device).unsqueeze(0)
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    return classes[pred.item()]
    
    
# url = 'https://vetstreet.brightspotcdn.com/dims4/default/60b6ebe/2147483647/thumbnail/590x420/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2F1b%2F10%2F187d796e44db9b3262b45d2e5aac%2Fdog-makes-eye-contact-thinkstockphotos-511375254-590lc031616.jpg'
# response = requests.get(url, stream=True)
# img = Image.open(response.raw)
#
# img=Image.open('dog.jpg')
# img = transform(img)
# image=img.to(device).unsqueeze(0)
# print(guessTheImage(image))