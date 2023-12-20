import os  # os模块，用来处理文件和目录
import torch
import torch.nn as nn
from PIL import Image  # 用来处理图像数据
from torch.utils.data import DataLoader  # 加载数据集
from torchvision import transforms  # 图像转换
from torchvision.transforms import ToTensor  # 可以将PIL图像或者numpy，ndarray，转换成torch.Tensor，

# 并且自动将图像的像素值归一化到[0，1]区间

to_tensor = ToTensor()  # 创建ToTensor实例，可以将图像数据转换成torch.Tensor

# 定义字典，存储不同数据集的图像转换操作
data_trasform = {
    'train': transforms.Compose([  # 用于组合多个图像变换操作
        transforms.Resize((64, 64)),  # 使用resize函数将图像大小调整到（64，64），然后将图像转换成torch.Tensor
        transforms.ToTensor()  # 然后将图像转换成torch.Tensor
    ]),
    'test': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
}


def load_Image_information(path):
    image_Root_Dir = "./"  # 设置图像的根目录为当前目录
    image_Dir = os.path.join(image_Root_Dir, path)  # 使用os.path.join将图片的根目录和path路径拼接起来，得到完整的图片路径
    return Image.open(image_Dir).convert('RGB')


class my_data_Set(nn.Module):
    def __init__(self, txt, trainsform=None, loader=None):  # 初始化方法，接受txt（文件路径），
        # trainsform（数据预处理方式，默认为None）和loader（数据加载器）默认为None
        super(my_data_Set, self).__init__()  # 调用父类初始化方法
        fp = open(txt, "r")
        images = []  # 创建一个空列表用于存储图像名称
        labels = []  # 创建一个空列表用于存储标签名称
        for line in fp:
            line.strip('\n')  # 去除行尾的换行符
            line.rstrip()  # 去除行首和行尾的空白字符
            information = line.split()  # 将行分割成多个部分
            images.append(information[0])  # 将图片名称添加到images列表中
            labels.append([float(i) for i in information[1:len(information)]])  # 添加标签到labels列表中
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


# 使用自定义的my_data_set函数加载数据
train_dateset = my_data_Set(r'./trainlabels.txt', trainsform=data_trasform['train'], loader=load_Image_information)
test_dateset = my_data_Set(r'./testlabels.txt', trainsform=data_trasform['test'], loader=load_Image_information)

# 创建数据加载器，设置批量大小为1
train_transform = DataLoader(train_dateset, batch_size=1, shuffle=True)
test_transform = DataLoader(test_dateset, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(  # 定义一个Sequential模型
            nn.Conv2d(3, 8, 3, 1),  # 8*62*62  第一个卷积层，输入通道3，输出通道8，卷积核大小3，步长1
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8*31*31  最大池化层  池化核大小为2
            nn.Conv2d(8, 16, 3, 1),  # 16*29*29   第二个卷积层
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16*14*14    池化层核心大小为2
            nn.Conv2d(16, 32, 3, 1),  # 32*12*12
            nn.ReLU(),
            nn.Flatten(),  # Flatten层，将多维输入一维化，常用在卷积层到全连接层的过渡
            nn.Linear(32 * 12 * 12, 128),  # 全连接层，输入维度为32*12*12，输出维度为128
            nn.ReLU(),
            nn.Dropout(),  # Dropout层，随机失活一部分神经元，防止过拟合
            nn.Linear(128, 7),
            nn.Sigmoid()  # Sigmoid激活函数，将输出转换为0到1之间的概率值
        )

    def forward(self, x):
        out = self.model(x)  # 通过模型进行前向计算，得到输出out
        return out


net = Net()  # 创建神经网络模型实例
loss_fn = nn.MSELoss()  # 损失函数
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001)

test_data_size = len(test_dateset)  # 计算测试集大小，即测试集包含的数据条数

epoch = 10  # 训练轮数
total_train_step = 0  # 初始化总的训练步长
test_test_step = 0  # 初始化测试步长

for i in range(epoch):
    print('-----------第{}轮训练开始----------'.format(i + 1))
    net.train()  # 开始训练网络
    for data in train_transform:  # 遍历所有训练数据
        imgs, lables = data  # 获取输入和标签
        # print(imgs.shape)
        output = net(imgs)  # 通过神经网络得到输出
        loss = loss_fn(output, lables)  # 计算损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数

    print(f'训练次数：{i + 1}, Loss: {loss}')

    net.eval()  # 将神经网络模型设置为评估模型
    total_test_loss = 0  # 初始化总的损失为0
    # total_accuracy_loss = 0
    with torch.no_grad():
        for data in test_transform:  # 遍历所有测试数据集
            imgs, lables = data  # 获取输入和标签
            output = net(imgs)  # 通过神经网络得到输出
            loss = loss_fn(output, lables)  # 计算损失
            total_test_loss = total_test_loss + loss  # 累加到总的损失中
            # accuracy = (output.argmax(1) == lables).sum()
            # total_accuracy_loss = total_accuracy_loss + accuracy

        print(f'整体测试集上的LOSS:{total_test_loss / test_data_size}')
        # print(f'整体测试集上的正确率accuracy:{total_accuracy_loss/test_data_size}')
