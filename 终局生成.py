import random
from game3 import GobangBoard as Board
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


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


def data(n):
    boards = []
    labels = []

    for i in range(n):
        board = Board()
        while True:
            # 获取可落子点
            valid_moves = board.get_valid_moves()
            move = random.choice(valid_moves)
            x = board.move(move)
            if x[1]:
                break
        print("{0}".format(i / n))
        boards.append(board.board)
        labels.append(board.winner)
        labels.append(-board.winner)
    print(0 in labels)
    boards = change(boards)
    print(len(boards),len(labels))
    return ChessDataset(boards,labels)


