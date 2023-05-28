import numpy as np


class Board(object):
    def __init__(self):
        self._board_weight = 14
        self._board_height = 14
        self._board = np.zeros(self._board_height * self._board_weight)
        self._current_player = -1
        self._available = list(range(self._board_height * self._board_weight))
        pass

    def location_to_move(self, location):
        return location[0] * self._board_weight + location[1]

    def move_to_location(self, move):
        return move // self._board_weight, move % self._board_height

    def move(self, place):
        if self._current_player == 1:
            self._board[place] = -1
            self._current_player = -1
            self._available.remove(place)
        else:
            self._board[place] = 1
            self._current_player = 1
            self._available.remove(place)

    def win(self, move):
        if move:
            if len(self._available) == 0:
                return True, 0
            _location = self.move_to_location(move)
            _top_border = max(0, _location[0]-4)
            _bottom_border = min(self._board_height, _location[0]+4)
            _left_border = max(0, _location[1]-4)
            _right_border = min(self._board_weight, _location[1]+4)

            _count = 0
            for _i in range(move-1, _left_border+_location[0]*self._board_weight-1, -1):
                if self._board[_i] == self._current_player:
                    _count += 1
                else:
                    break
            for _i in range(move+1, _right_border+_location[0]*self._board_weight+1, 1):
                if self._board[_i] == self._current_player:
                    _count += 1
            if _count >= 4:
                return True, self._current_player

            _count = 0
            for _i in range(move-self._board_weight, _top_border*self._board_weight+_location[1]-self._board_weight,
                            -self._board_weight):
                if self._board[_i] == self._current_player:
                    _count += 1
                else:
                    break
            for _i in range(move+self._board_weight, _bottom_border*self._board_weight+_location[1]+self._board_weight,
                            self._board_weight):
                if self._board[_i] == self._current_player:
                    _count += 1
            if _count >= 4:
                return True, self._current_player

            _count = 0
            for _i in range(move - self._board_weight - 1, _top_border * self._board_weight + _left_border-self._board_weight-1,
                            -self._board_weight-1):
                if self._board[_i] == self._current_player:
                    _count += 1
                else:
                    break
            for _i in range(move + self._board_weight + 1, _bottom_border * self._board_weight + _right_border + self._board_weight + 1,
                            self._board_weight+1):
                if self._board[_i] == self._current_player:
                    _count += 1
            if _count >= 4:
                return True, self._current_player

            _count = 0
            for _i in range(move - self._board_weight + 1,
                            _top_border * self._board_weight + _right_border - self._board_weight + 1,
                            -self._board_weight + 1):
                if self._board[_i] == self._current_player:
                    _count += 1
                else:
                    break
            for _i in range(move + self._board_weight - 1,
                            _bottom_border * self._board_weight + _left_border + self._board_weight - 1,
                            self._board_weight - 1):
                if self._board[_i] == self._current_player:
                    _count += 1
            if _count >= 4:
                return True, self._current_player
            return False, 0
        else:
            return False, 0


if __name__ == '__main__':
    pass
