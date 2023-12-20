# File: main.py 
# Auther: Wang Yu
# Time: 2022/11/26 t9:02
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import ToTensor

to_tensor = ToTensor()

data_trasform = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
}

def load_Image_information(path):
    image_Root_Dir = "./imgdata"
    image_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(image_Dir).convert('RGB')

class my_data_Set(nn.Module):
    def __init__(self, txt, trainsform=None, loader=None):
        super(my_data_Set, self).__init__()
        fp = open(txt, "r")
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(i) for i in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = trainsform
        self.loader = loader

    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]

        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        # image = to_tensor(image)
        label = torch.FloatTensor(label)

        return image, label

    def __len__(self):
        return len(self.images)

train_dateset = my_data_Set(r'./train.txt', trainsform=data_trasform['train'], loader=load_Image_information)
test_dateset = my_data_Set(r'./test.txt', trainsform=data_trasform['test'], loader=load_Image_information)
train_transform = DataLoader(train_dateset, batch_size=1, shuffle=True)
test_transform = DataLoader(test_dateset, batch_size=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1),  # 8*62*62
            nn.ReLU(),
            nn.MaxPool2d(2),   # 8*31*31
            nn.Conv2d(8, 16, 3, 1),  # 16*29*29
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16*14*14
            nn.Conv2d(16, 32, 3, 1),  # 32*12*12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*12*12, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.model(x)

        return out

net = Net()
loss_fn = nn.L1Loss()
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

test_data_size = len(test_dateset)

epoch = 10
total_train_step = 0
test_test_step = 0
for i in range(epoch):
    print('-----------第{}轮训练开始----------'.format(i+1))
    net.train()
    for data in train_transform:
        imgs, lables = data
        # print(imgs.shape)
        output = net(imgs)
        loss = loss_fn(output, lables)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'训练次数：{i+1}, Loss: {loss}')

    net.eval()
    total_test_loss = 0
    # total_accuracy_loss = 0
    with torch.no_grad():
        for data in test_transform:
            imgs, lables = data
            output = net(imgs)
            loss = loss_fn(output, lables)
            total_test_loss = total_test_loss + loss
            # accuracy = (output.argmax(1) == lables).sum()
            # total_accuracy_loss = total_accuracy_loss + accuracy

        print(f'整体测试集上的LOSS:{total_test_loss/test_data_size}')
        # print(f'整体测试集上的正确率accuracy:{total_accuracy_loss/test_data_size}'

