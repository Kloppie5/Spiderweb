
import tkinter as tk

class NetworkRenderer:

    def __init__(self, network):
        self.network = network

        self.width = 800
        self.height = 600

        self.node_radius = 20
        self.layer_spacing = 200
        self.neuron_spacing = 80

    def render(self):
        root = tk.Tk()
        root.title("Neural Network")

        canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        canvas.pack()

        positions = self._calculate_positions()

        self._draw_connections(canvas, positions)
        self._draw_neurons(canvas, positions)

        root.mainloop()

    def _calculate_positions(self):
        positions = []

        num_layers = len(self.network.layers)

        for i, layer in enumerate(self.network.layers):
            layer_positions = []

            x = (i + 1) * self.width / (num_layers + 1)

            num_neurons = len(layer.neurons)

            for j in range(num_neurons):
                y = (j + 1) * self.height / (num_neurons + 1)
                layer_positions.append((x, y))

            positions.append(layer_positions)

        return positions
    
    def _draw_connections(self, canvas, positions):
        for i in range(len(positions) - 1):
            for (x1, y1) in positions[i]:
                for (x2, y2) in positions[i + 1]:
                    canvas.create_line(x1, y1, x2, y2)

    def _draw_neurons(self, canvas, positions):
        for layer in positions:
            for (x, y) in layer:
                r = self.node_radius
                canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightblue")
