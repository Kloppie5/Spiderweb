
class TicTacToe:

    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def make_move(self, idx):
        if self.board[idx] != 0:
            return False

        self.board[idx] = self.current_player
        self.current_player *= -1
        return True

    def check_winner(self):
        board = self.board
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]

        for a,b,c in wins:
            if board[a] == board[b] == board[c] != 0:
                return board[a]

        if 0 not in board:
            return 0

        return None