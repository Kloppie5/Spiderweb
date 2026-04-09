import os

from nn.activations import sigmoid, sigmoid_derivative
from nn.layer import Layer
from nn.network import Network
from nn.neuron import Neuron
from nn.persistence import save_network, load_network

from ai.tic_tac_toe_agent import Agent
from training.tic_tac_toe_trainer import Trainer

from visualizations.network_renderer import NetworkRenderer

MODEL_PATH = "./models/tic_tac_toe_model.json"

if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    net = load_network(
        MODEL_PATH,
        neuron_class=Neuron,
        layer_class=Layer,
        network_class=Network,
        activation=sigmoid,
        activation_derivative=sigmoid_derivative
    )
else:
    print("No model found. Creating new network...")
    # 9->18
    l1 = Layer([
        Neuron([0.1]*9, 0.0, sigmoid, sigmoid_derivative)
        for _ in range(18)
    ])
    #18->9
    l2 = Layer([
        Neuron([0.1]*18, 0.0, sigmoid, sigmoid_derivative)
        for _ in range(9)
    ])

    net = Network([l1, l2])

agent = Agent(net)
trainer = Trainer(agent)

wins = 0
losses = 0
draws = 0

game = 0
win_rate = 0.0
while win_rate < 82.0:
    result, history = trainer.play_game()
    trainer.train_from_game(history, result)
    game += 1

    if result == 1:
        wins += 1
    elif result == -1:
        losses += 1
    else:
        draws += 1

    if game % 100 == 0:

        win_rate = wins / game * 100
        loss_rate = losses / game * 100
        draw_rate = draws / game * 100

        print(f"After {game} games:")
        print(f"  Wins:   {wins} ({win_rate:.1f}%)")
        print(f"  Losses: {losses} ({loss_rate:.1f}%)")
        print(f"  Draws:  {draws} ({draw_rate:.1f}%)")
        print("-" * 30)
    

save_network(net, MODEL_PATH)
print("Training done")
