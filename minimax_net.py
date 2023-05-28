import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from 终局生成 import data
import numpy as np


class GobangNet(nn.Module):
    def __init__(self):
        super(GobangNet, self).__init__()
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 15 * 15, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 15 * 15)
        x = self.fc(x)
        return x


def train(net, dataloader, epoches, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epoches):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels.float().unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))


if __name__ == '__main__':
    while 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = data(100000)
        dataloader = DataLoader(dataset, batch_size=128)
        # 创建模型对象
        net = torch.load('0.pth').to(device)
        train(net, dataloader, epoches=5, lr=0.001)
        torch.save(net,'0.pth')
