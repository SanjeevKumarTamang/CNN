import os
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
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

def guessTheImage(image):
    model = LeNet().to(device)
    pth = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(pth + '/cnn_cat_dog.pth'))
    model.eval()
    img = Image.open(image)
    img = transform(img)
    image = img.to(device).unsqueeze(0)
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    return classes[pred.item()]
