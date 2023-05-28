import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.Flatten(),
            nn.Linear(32 * 15 * 15, 120),
            nn.Linear(120, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


n = Net().to('cuda')


def net(x):
    return n(x)

if __name__ == '__main__':
    board = torch.randn(1,3,15,15).to('cuda')
    print(net(board))
