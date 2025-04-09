import os.path

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt


'''
导入数据集
    root表示数据集根目录路径
    train=True 代表读入的数据作为训练集（如果为true则从training.pt创建数据集，否则从test.pt创建数据集）
    train=False 代表读入的数据作为测试集（如果为true则从training.pt创建数据集，否则从test.pt创建数据集）
    transform则是读入我们自己定义的数据预处理操作
    download=True则是当我们的根目录（root）下没有数据集时，便自动下载
'''
train_data = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


'''
加载 MNIST 训练和测试数据集
batch_size为64
'''
batch_size = 64

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)


# matplotlib展示MNIST图像
plt.figure(figsize=(8, 8))
iter_dataloader = iter(test_dataloader)

n = 1

# 取出n*batch_size张图片可视化
for i in range(n):
    images, labels = next(iter_dataloader)
    image_grid = torchvision.utils.make_grid(images)
    plt.subplot(1, n, i+1)
    plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))

    # 是否使用GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

# 搭建LeNet网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积神经网络
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )
        # 分类器
        self.fc  = nn.Sequential(
            nn.Linear(576, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10) # 输出为10类
        )

    def forward(self, x):
        out = self.conv(x)  # 输出 16*5*5 特征图
        out = out.view(out.size(0), -1)     # 展平（1， 16*5*5）
        out = self.fc(out)  # 输出 10
        return out

def train(network):
    losses = []
    iteration = 0

    epochs = 10

    for epoch in range(epochs):
        loss_sum = 0
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = network(X)
            loss = loss_fn(pred, y)

            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = loss_sum / len(train_dataloader.dataset)
        losses.append(mean_loss)
        iteration += 1
        print(f"Epoch {epoch+1} loss: {mean_loss:>7f}")

    # 训练完毕保存最后一轮训练的模型
    torch.save(network.state_dict(), "model.pth")

    # 绘制损失函数曲线
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.plot(list(range(iteration)), losses)


# 初始化网络
network = LeNet()
network.to(device)


# 使用交叉熵损失函数
# 优化器为随机梯度下降算法作
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=network.parameters(), lr=0.001, momentum=0.9)

# 若模型已存在，加载已经预训练的模型；否则开始训练过程
if os.path.exists('model.pth'):
    network.load_state_dict(torch.load('model.pth'))
else:
    train(network)


# 在测试集上进行验证
positive = 0
negative = 0
# 循环遍历测试集中的所有示例
for X, y in test_dataloader:
    with torch.no_grad():
        # 把数据和标签发送到设备
        X, y = X.to(device), y.to(device)
        # 通过模型前向传递数据
        pred = network(X)
        for item in zip(pred, y):
            if torch.argmax(item[0]) == item[1]:
                positive += 1
            else:
                negative += 1
acc = positive / (positive + negative)
print(f"Accuracy: {acc * 100}%")


# 定义扰动值列表
eps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# 寻找对抗样本，并可视化
# X 为图像数据，y为标签
for X, y in test_dataloader:
    X, y = X.to(device), y.to(device)

    # 设置张量的requires_grad属性，在调用backward()方法时反向传播计算梯度，这对于攻击很关键
    X.requires_grad = True
    # 通过模型前向传递数据
    pred = network(X)
    # 将所有现有的渐变归零
    network.zero_grad()
    # 计算损失
    loss = loss_fn(pred, y)
    # 计算后向传递模型的梯度
    loss.backward()

    plt.figure(figsize=(15, 8))
    # 输出原始数据梯度的元素符号
    plt.subplot(121)
    image_grid = torchvision.utils.make_grid(torch.clamp(X.grad.sign(), 0, 1))
    plt.imshow(np.transpose(image_grid.cpu().numpy(), (1, 2, 0)))

    '''
    开始FGSM攻击：通过调整输入图像的每个像素来创建扰动图像
    X.grad.sign()：获取数据梯度的元素符号
    eps： 待添加的扰动值列表
    X_adv： 添加扰动后的图片
    '''
    X_adv = X + eps[2] * X.grad.sign()
    # 将添加扰动后的张量中的每个元素范围限制在[0,1]之间
    X_adv = torch.clamp(X_adv, 0, 1)

    # 输出添加扰动后的图片
    plt.subplot(122)
    image_grid = torchvision.utils.make_grid(X_adv)
    plt.imshow(np.transpose(image_grid.cpu().numpy(), (1, 2, 0)))
    plt.show()

    break

# 用对抗样本替代原始样本，测试准确度
# 探究不同epsilon对LeNet分类准确度的影响
positive = 0
negative = 0
acc_list = []
for epsilon in eps:
    positive = 0
    negative = 0
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)

        X.requires_grad = True
        pred = network(X)
        network.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()

        X = X + epsilon * X.grad.sign()
        X_adv = torch.clamp(X, 0, 1)

        # X_adv代表生成的对抗样本，用来检测模型的分类情况
        pred = network(X_adv)
        for item in zip(pred, y):
            if torch.argmax(item[0]) == item[1]:
                positive += 1
            else:
                negative += 1

    acc = positive / (positive + negative)
    print(f"epsilon={epsilon} acc: {acc * 100}%")
    acc_list.append(acc)

plt.rcParams['axes.facecolor'] = 'white'
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot(eps, acc_list, marker='o')
plt.savefig("acc.png")