
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
        for i, layer in enumerate(self.network.layers[:-1]):
            next_layer = self.network.layers[i + 1]

            for j, neuron in enumerate(layer.neurons):
                for k, next_neuron in enumerate(next_layer.neurons):
                    x1, y1 = positions[i][j]
                    x2, y2 = positions[i + 1][k]

                    weight = next_neuron.weights[j]

                    color = self._weight_to_color(weight)
                    width = self._weight_to_width(weight)

                    canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _weight_to_color(self, w):
        if w > 0:
            intensity = min(255, int(abs(w) * 255))
            return f"#00{intensity:02x}00"
        elif w < 0:
            intensity = min(255, int(abs(w) * 255))
            return f"#{intensity:02x}0000"
        else:
            return "#cccccc"

    def _weight_to_width(self, w):
        return max(2, abs(w) * 10)

    def _draw_neurons(self, canvas, positions):
        for layer in positions:
            for (x, y) in layer:
                r = self.node_radius
                canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightblue")
