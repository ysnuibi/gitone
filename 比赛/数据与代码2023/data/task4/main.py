import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

to_tensor = ToTensor()

data_transform = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

}


def load_image_information(path):
    image_Root_Dir = './'
    image_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(image_Dir).convert('RGB')


class my_data_Set(nn.Module):
    def __init__(self, txt, transform=None, loader=None):
        super(my_data_Set, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.split('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(i) for i in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)

        return image, label

    def __len__(self):
        return len(self.images)


train_dataset = my_data_Set(r'./trainlabels.txt', transform=data_transform['train'], loader=load_image_information)
test_dataset = my_data_Set(r'./testlabels.txt', transform=data_transform['test'], loader=load_image_information)

train_transform = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_transform = DataLoader(test_dataset, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(128, 7)

        )

    def forward(self, x):
        out = self.model(x)
        return out


net = Net()
loss_fn = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001)

epoch = 10
test_data_size = len(test_dataset)

total_train_step = 0
test_test_step = 0

for i in range(epoch):
    print('--------第{}轮训练开始-------'.format(i + 1))
    net.train()
    for data in train_transform:
        imgs, lables = data
        outputs = net(imgs)
        loss = loss_fn(outputs, lables)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'训练轮数：{i + 1}，Loss：{loss}')

    net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_transform:
            imgs, lables = data
            outputs = net(imgs)
            loss = loss_fn(outputs, lables)
            total_test_loss = total_test_loss + loss
        print(f'整体测试集上的LOSS：{total_test_loss / test_data_size}')

# 总结：1.定义字典，存储图像转换格式
#      2.加载图像信息
#      3.自定义数据转换格式
#      4.加载数据
#      5.自定义神经网络模型
#      6.常见神经网络模型实例，损失函数，优化器，训练轮数。。。
#      7.训练与评估:   训练数据，获取输入和标签，通过神经网络得到输出，计算损失，清空梯度，反向传播，更新参数
