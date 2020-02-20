import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10,scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_dataset = datasets.ImageFolder(root='../dataset_dogs_vs_cats/train', transform=transform_train)
validation_dataset = datasets.ImageFolder(root='../dataset_dogs_vs_cats/test', transform=transform)

# training_dataset = datasets.ImageFolder(root='dataset_dogs_vs_cats/train', transform=transform_train)
# validation_dataset = datasets.ImageFolder(root='dataset_dogs_vs_cats/test', transform=transform)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)  # swapping the axes
    # print(image.shape)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

classes=('cat','dog')

dataiter = iter(training_loader)
images, labels = dataiter.next()
# fig = plt.figure(figsize=(20, 4))
#
# for i in np.arange(10):
#     ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
#     plt.imshow(im_convert(images[i]))
#     ax.set_title(classes[labels[i].item()]+str(labels[i].item()))
# plt.show()

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

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 30
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    for inputs, labels in training_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        # inputs = inputs.view(inputs.shape[0], -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs=val_inputs.to(device)
                val_labels=val_labels.to(device)
                # val_inputs = val_inputs.view(val_inputs.shape[0], -1)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += torch.sum(val_preds == val_labels.data)
                val_running_loss += val_loss.item()

        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.float() / len(training_loader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)

        print('epoch', e + 1)
        print('training loss :{:.4f},acc {:.4f}'.format(epoch_loss, epoch_acc.item()))
        print('Validation loss :{:.4f},acc {:.4f}'.format(val_epoch_loss, val_epoch_acc.item()))

torch.save(model.state_dict(), 'cnn_cat_dog.pth')
plt.plot(running_corrects_history, label="training accuracy")
plt.plot(val_running_corrects_history, label="validation accuracy")
plt.legend()
plt.show()

# dataiter = iter(validation_loader)
# images, labels = dataiter.next()
# images=images.to(device)
# labels=labels.to(device)
# # images_ = images.view(images.shape[0], -1)
# output = model(images)
# _, preds = torch.max(output, 1)
#
# fig = plt.figure(figsize=(15, 4))
#
# for i in np.arange(5):
#     ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
#     plt.imshow(im_convert(images[i]))
#     ax.set_title("{} ({})".format(str(classes[preds[i].item()]), str(classes[labels[i].item()])),
#                  color=("green" if preds[i] == labels[i] else "red"))
# plt.show()

import requests
from PIL import Image
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcToQSNcJtdWfAkZEiB-2GYzNGZGWEy4fIZliWUfrJKtWo27-lU5'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
# plt.imshow(img)

img = transform(img)
# plt.imshow(im_convert(img))

# img = img.view(img.shape[0], -1)
image=img.to(device).unsqueeze(0)
outputs = model(image)

_, pred = torch.max(outputs, 1)
print(classes[pred.item()])