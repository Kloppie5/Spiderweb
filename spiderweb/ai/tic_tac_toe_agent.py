import random

class Agent:

    def __init__(self, network):
        self.network = network

    def encode_board(self, board):
        return board[:]

    def choose_move(self, board, legal_moves):
        inputs = self.encode_board(board)

        outputs = self.network.forward(inputs)

        for i in range(9):
            if i not in legal_moves:
                outputs[i] = -999

        best = max(range(9), key=lambda i: outputs[i])
        return best
