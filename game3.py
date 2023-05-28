import numpy as np
import random


class GobangBoard(object):
    def __init__(self, height=15, width=15, n_in_row=5, board_in_=None,board_in=None):
        if not board_in_:
            self.height = int(height)
            self.width = int(width)
            self.n_in_row = n_in_row
            self.board: np.ndarray = np.zeros((height, width), dtype=int)
            self.player = 1
            self.last_move = None
            self.game_over = False
            self.valid_moves = [(i, j) for i in range(height) for j in range(width)]
            self.winner = None
        else:
            self.board: np.ndarray = board_in
            self.height = len(board_in)
            self.width = len(board_in[0])
            self.n_in_row = n_in_row
            self.player = 1
            self.last_move = None
            self.game_over = False
            self.winner = None
            self.valid_moves = []
            for i in range(height):
                for j in range(width):
                    if self.board[i][j] == 0:
                        self.valid_moves.append((i, j))

    def move(self, location):
        x, y = location
        if x < 0 or x >= self.height or y < 0 or y >= self.width or self.board[x][y] != 0:
            return False, False
        self.board[x][y] = self.player
        self.last_move = location
        if self.is_win():
            self.game_over = True
            self.winner = self.player
            return True, True
        elif self.valid_moves:
            self.player = - self.player
            self.valid_moves.remove(location)
        else:
            self.game_over = True
            self.winner = 0
            return True, True
        return True, False

    def get_valid_moves(self):
        return self.valid_moves

    def is_win(self):
        if self.last_move is None:
            return False
        x, y = self.last_move
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in dirs:
            count1, count2 = 1, 1
            for i in range(1, self.n_in_row):
                if y + i * dy < 0 or y + i * dy >= self.width or \
                        x + i * dx < 0 or x + i * dx >= self.height or \
                        self.board[x + i * dx][y + i * dy] != self.player:
                    break
                count1 += 1
            for i in range(1, self.n_in_row):
                if y - i * dy < 0 or y - i * dy >= self.width or \
                        x - i * dx < 0 or x - i * dx >= self.height or \
                        self.board[x - i * dx][y - i * dy] != self.player:
                    break
                count2 += 1
            if count1 + count2 - 1 >= self.n_in_row:
                return True
        return False

    def print_board(self):
        print('　', end='')
        for x in range(self.width):
            print("{0:2d}".format(x), end=' ')
        print()
        for i in range(self.height):
            print("{0:2d}".format(i), end='　')
            for j in range(self.width):
                if self.board[i][j] == 1:
                    print('X', end='　')
                elif self.board[i][j] == -1:
                    print('O', end='　')
                else:
                    print('_', end='　')
            print()

    def show_last_move(self):
        if self.last_move is not None:
            x, y = self.last_move
            print('Last move: ({0}, {1})'.format(x, y))
            print()


def policy(board_):
    valid_moves = board_.get_valid_moves()
    move = random.choice(valid_moves)
    return move


if __name__ == '__main__':
    board = GobangBoard()
    board.move((0, 0))
