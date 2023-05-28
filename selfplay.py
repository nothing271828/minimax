import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from game3 import GobangBoard as Board
from minimax_search import MinimaxPlayer

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
])


class GobangNet(nn.Module):
    def __init__(self):
        super(GobangNet, self).__init__()
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(9,9), padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(9,9), padding=4),
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


def change(boards_):
    boards_out = []
    for board_c in boards_:
        board_out1, board_out2 = np.zeros((3, 15, 15), dtype=np.float64), np.zeros((3, 15, 15), dtype=np.float64)
        for i in range(15):
            for j in range(15):
                if board_c[i][j] == 1:
                    board_out1[0][i][j] = 1.0  # 黑子位置为 (i, j)，在第 0 个通道上赋值 1.0
                    board_out2[1][i][j] = 1.0
                elif board_c[i][j] == -1:
                    board_out1[1][i][j] = 1.0  # 白子位置为 (i, j)，在第 1 个通道上赋值 1.0
                    board_out2[0][i][j] = 1.0
                else:
                    board_out1[2][i][j] = 1.0
                    board_out2[2][i][j] = 1.0
        boards_out.append(board_out1)
        boards_out.append(board_out2)

    return torch.stack([torch.from_numpy(board).float() for board in boards_out])


def evaluate(board, net_in):
    board_tensor = np.zeros((3, 15, 15), dtype=np.float32)
    for i in range(15):
        for j in range(15):
            if board[i][j] == 1:
                board_tensor[0][i][j] = 1.0  # 黑子位置为 (i, j)，在第 0 个通道上赋值 1.0
            elif board[i][j] == -1:
                board_tensor[1][i][j] = 1.0  # 白子位置为 (i, j)，在第 1 个通道上赋值 1.0
            else:
                board_tensor[2][i][j] = 1.0
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
    board_tensor = board_tensor.unsqueeze(0)

    with torch.no_grad():
        output = net_in(board_tensor)
    return output


def self_play():
    board_list = []
    board = Board()
    player1 = MinimaxPlayer(depth=1, policy=evaluate1, train_mod=0)
    player2 = MinimaxPlayer(depth=1, policy=evaluate2, train_mod=0)
    while True:
        board1 = copy.deepcopy(board)
        move = player1.get_action(board1)
        x = board.move(move)
        board_list.append(copy.deepcopy(board.board))
        if x[1]:
            break
        board_list.append(board.board)
        board2 = copy.deepcopy(board)
        move = player2.get_action(board2)
        x = board.move(move)
        board_list.append(copy.deepcopy(board.board))
        if x[1]:
            break
        board_list.append(board.board)
    board.print_board()
    # print(board_list[0],'test')
    return board_list, board.player


class ChessDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)


def train(net, dataloader, epoches, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epoches):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            # print(inputs.shape, inputs.dtype, labels, labels.shape)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels.float().unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))
        if running_loss == 0:
            break


# ChessDataset(boards, labels,
#                 transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while 1:
        net1 = torch.load('2.1.pth').to(device)
        net1.eval()
        net2 = torch.load('2.2.pth').to(device)
        net2.eval()


        def evaluate1(board):
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
                output = net1(board_tensor)
            return output


        def evaluate2(board):
            board_tensor = np.zeros((3, 15, 15), dtype=np.float32)
            for i in range(15):
                for j in range(15):
                    if board[i][j] == -1:
                        board_tensor[0][i][j] = 1.0  # 黑子位置为 (i, j)，在第 0 个通道上赋值 1.0
                    elif board[i][j] == 1:
                        board_tensor[1][i][j] = 1.0  # 白子位置为 (i, j)，在第 1 个通道上赋值 1.0
                    else:
                        board_tensor[2][i][j] = 1.0
            board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
            board_tensor = board_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = net2(board_tensor)
            return output


        # boards = torch.load('data.pt')
        n = 4
        win_count = 0
        for main_i in range(n):
            boards, winner = self_play()
            print('finish self-play', "winner:", winner)
            boards = change(boards)
            labels = [winner if i % 2 == 0 else -winner for i in range(len(boards))]
            print(len(boards))
            if main_i == 0:
                final_boards = boards
                final_labels = labels
            else:
                final_boards = torch.cat((final_boards, boards), dim=0)
                final_labels = final_labels+labels
            if winner == 1:
                win_count += 1
        rate = win_count / n
        print("胜率:", rate)
        torch.save(final_boards, 'data.pt')
        dataset = ChessDataset(final_boards, final_labels)
        dataloader = DataLoader(dataset, batch_size=8)


        if rate <= 0.5:
            net = torch.load('2.1.pth').to(device).train()
        else:
            net = torch.load('2.2.pth').to(device).train()
        train(net, dataloader, 5, 0.001)
        if rate <= 0.5:
            torch.save(net, '2.1.pth')
        else:
            torch.save(net, '2.2.pth')
