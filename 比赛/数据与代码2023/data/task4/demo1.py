import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder

# 定义卷积神经网络模型
class FaceKeyPointNet(nn.Module):
    def __init__(self):
        super(FaceKeyPointNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
# 数据加载
train_dataset = ImageFolder('imgdata/train', transform=ToTensor())
test_dataset = ImageFolder('imgdata/test', transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建模型和优化器
model = FaceKeyPointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()  # 使用L1损失函数

# 训练和测试
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
    # 在测试集上评估模型性能
    with torch.no_grad():
        total_loss = 0.0
        total_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            total_samples += len(labels)
        avr_loss = total_loss / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avr_loss:.4f}')
# 保存模型
torch.save(model.state_dict(), 'facekeypointnet.pth')