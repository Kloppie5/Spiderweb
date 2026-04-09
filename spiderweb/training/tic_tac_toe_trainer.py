import random
from games.tic_tac_toe import TicTacToe

class Trainer:

    def __init__(self, agent):
        self.agent = agent

    def play_game(self):
        game = TicTacToe()
        history = []

        while True:
            board = game.board[:]
            moves = game.available_moves()

            if game.current_player == 1:
                move = self.agent.choose_move(board, moves)
            else:
                move = random.choice(moves)

            game.make_move(move)
            history.append((board, move))

            winner = game.check_winner()
            if winner is not None:
                return winner, history

    def train_from_game(self, history, result):
        lr = 0.1

        reward = 1 if result == 1 else -1 if result == -1 else 0

        output_layer = self.agent.network.layers[-1]

        for board, move in history:
            outputs = self.agent.network.forward(board)

            target = [0.0] * len(outputs)
            target[move] = 1.0

            errors = [(outputs[i] - target[i]) * reward for i in range(len(outputs))]

            self.agent.network.backward(errors, lr)
