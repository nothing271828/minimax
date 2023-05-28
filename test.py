from game3 import GobangBoard as Board
import numpy as np
import torch
import torch.nn as nn


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
            nn.Tanh()  # 加入归一化层
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 15 * 15)
        x = self.fc(x)
        return x


def evaluate(board):
    board_tensor = np.zeros((3, 15, 15), dtype=np.float32)
    for i_1 in range(15):
        for j in range(15):
            if board[i_1][j] == 1:
                board_tensor[0][i_1][j] = 1.0  # 黑子位置为 (i, j)，在第 0 个通道上赋值 1.0
            elif board[i_1][j] == -1:
                board_tensor[1][i_1][j] = 1.0  # 白子位置为 (i, j)，在第 1 个通道上赋值 1.0
            else:
                board_tensor[2][i_1][j] = 1.0
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
    board_tensor = board_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(board_tensor)
    return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load('2.2.pth').to(device)
    net.eval()
    board = [
        [-1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    board = np.array(board)
    board = Board(board_in_=True,board_in=board)
    board.print_board()
    print(evaluate(board.board))
