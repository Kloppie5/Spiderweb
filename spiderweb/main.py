import os

from nn.activations import step
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
        activation=step
    )
else:
    print("No model found. Creating new network...")
    layer = Layer([
        Neuron([0.1]*9, 0.0, step) for _ in range(9)
    ])

    net = Network([layer])

agent = Agent(net)
trainer = Trainer(agent)

for i in range(1000):
    result, history = trainer.play_game()
    trainer.train_from_game(history, result)

    if i % 100 == 0:
        print(f"Game {i}, result: {result}")

save_network(net, MODEL_PATH)
print("Training done")
