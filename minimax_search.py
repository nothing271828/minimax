import numpy as np
import random

class MinimaxPlayer:
    def __init__(self, policy, depth=3, train_mod=0):
        self.policy = policy
        self.depth = depth
        self.train_mod = train_mod

    def get_action(self, board):
        _, action = self._minimax(board, 0, -np.inf, np.inf)
        return action

    def _minimax(self, board, depth, alpha, beta):
        valid_moves = board.get_valid_moves()
        if depth == self.depth or board.game_over:
            return self.policy(board.board)[0][0] + random.random()*self.train_mod, None  # 返回当前局面的估值和动作为空

        if board.player == 1:  # 极大层
            max_val = -np.inf
            best_action = valid_moves[0]
            for move in valid_moves:
                board.move(move)
                v, _ = self._minimax(board, depth+1, alpha, beta)
                board.board[move] = 0
                board.player = -board.player
                if v > max_val:
                    max_val = v
                    best_action = move
                alpha = max(alpha, max_val)
                if beta <= alpha:
                    break
            return max_val, best_action

        else:  # 极小层
            min_val = np.inf
            worst_action = valid_moves[0]
            for move in valid_moves:
                board.move(move)
                v, _ = self._minimax(board, depth+1, alpha, beta)
                board.board[move] = 0
                board.player = -board.player
                if v < min_val:
                    min_val = v
                    worst_action = move
                beta = min(beta, min_val)
                if beta <= alpha:
                    break
            return min_val, worst_action