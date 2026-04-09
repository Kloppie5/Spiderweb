
import tkinter as tk

class NetworkRenderer:

    def __init__(self, network):
        self.network = network

        self.width = 800
        self.height = 600

        self.node_radius = 20
        self.layer_spacing = 200
        self.neuron_spacing = 80

        self.network_width = int(self.width * 0.6)
        self.graph_width = self.width - self.network_width

        self.history = []

        self.current_image = None
        self.current_label = None
        self.current_prediction = None
        self.current_outputs = None

    def start(self):
        self.root = tk.Tk()
        self.root.title("Neural Network Training")

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="white"
        )
        self.canvas.pack()

    def update(self):
        self.canvas.delete("all")

        positions = self._calculate_positions()

        self._draw_connections(self.canvas, positions)
        self._draw_neurons(self.canvas, positions)
        self._draw_graph(self.canvas)
        self._draw_mnist_panel(self.canvas)

        self.root.update_idletasks()
        self.root.update()

    def _calculate_positions(self):
        positions = []

        input_size = len(self.network.layers[0].neurons[0].weights)
        input_positions = []
        for j in range(input_size):
            x = self.network_width / (len(self.network.layers) + 2)
            y = (j + 1) * self.height / (input_size + 1)
            input_positions.append((x, y))
        positions.append(input_positions)

        num_layers = len(self.network.layers)
        for i, layer in enumerate(self.network.layers):

            x = (i + 2) * self.network_width / (num_layers + 2)

            num_neurons = len(layer.neurons)
            layer_positions = []
            for j in range(num_neurons):
                y = (j + 1) * self.height / (num_neurons + 1)
                layer_positions.append((x, y))
            positions.append(layer_positions)

        return positions
    
    def _draw_connections(self, canvas, positions):

        input_size = len(self.network.layers[0].neurons[0].weights)
        for j in range(input_size):
            x1, y1 = positions[0][j]

            for k, neuron in enumerate(self.network.layers[0].neurons):
                x2, y2 = positions[1][k]

                weight = neuron.weights[j]

                canvas.create_line(
                    x1, y1, x2, y2,
                    fill=self._weight_to_color(weight),
                    width=self._weight_to_width(weight)
                )

        for i, layer in enumerate(self.network.layers[:-1]):
            next_layer = self.network.layers[i + 1]

            for j, neuron in enumerate(layer.neurons):
                for k, next_neuron in enumerate(next_layer.neurons):
                    x1, y1 = positions[i + 1][j]
                    x2, y2 = positions[i + 2][k]

                    weight = next_neuron.weights[j]

                    color = self._weight_to_color(weight)
                    width = self._weight_to_width(weight)

                    canvas.create_line(
                        x1, y1,
                        x2, y2,
                        fill=color,
                        width=width
                    )

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
        for i, layer in enumerate(positions):
            for (x, y) in layer:
                r = self.node_radius

                if i == 0:
                    color = "lightgreen"
                else:
                    color = "lightblue"

                canvas.create_oval(
                    x - r, y - r,
                    x + r, y + r,
                    fill=color,
                    outline="black"
                )

    def _draw_graph(self, canvas):
        if len(self.history) < 2:
            return

        x_offset = self.network_width
        w = self.graph_width
        h = self.height

        max_points = 200
        data = self.history[-max_points:]

        for i in range(len(data) - 1):
            x1 = x_offset + (i / max_points) * w
            x2 = x_offset + ((i + 1) / max_points) * w

            y1 = h - data[i] * h
            y2 = h - data[i + 1] * h

            canvas.create_line(
                x1, y1,
                x2, y2,
                fill="green",
                width=2
            )

        canvas.create_text(
            x_offset + 60,
            20,
            text=f"{data[-1]*100:.1f}%",
            font=("Arial", 14, "bold"),
            fill="black"
        )
        canvas.create_text(
            x_offset + w / 2,
            40,
            text="Win Rate (last 100 games)",
            font=("Arial", 10),
            fill="gray"
        )
    
    def set_mnist_sample(self, image, label, outputs):
        self.current_image = image
        self.current_label = label
        self.current_outputs = outputs
        self.current_prediction = outputs.index(max(outputs))

    def _draw_mnist_panel(self, canvas):
        x_offset = self.network_width + 20
        y_offset = 50
        pixel_size = 8

        canvas.create_text(
            x_offset + 100,
            20,
            text="MNIST Input",
            font=("Arial", 12, "bold")
        )

        if self.current_image:
            for i in range(28):
                for j in range(28):
                    val = self.current_image[i * 28 + j]
                    gray = int(val * 255)

                    color = f"#{gray:02x}{gray:02x}{gray:02x}"

                    x1 = x_offset + j * pixel_size
                    y1 = y_offset + i * pixel_size

                    canvas.create_rectangle(
                        x1, y1,
                        x1 + pixel_size,
                        y1 + pixel_size,
                        fill=color,
                        outline=color
                    )

        if self.current_prediction is not None:
            canvas.create_text(
                x_offset + 100,
                300,
                text=f"Pred: {self.current_prediction}",
                font=("Arial", 14, "bold"),
                fill="blue"
            )

            canvas.create_text(
                x_offset + 100,
                330,
                text=f"Label: {self.current_label}",
                font=("Arial", 14),
                fill="green" if self.current_prediction == self.current_label else "red"
            )

        if self.current_outputs:
            base_y = 370

            for i, v in enumerate(self.current_outputs):
                bar_len = v * 100

                canvas.create_rectangle(
                    x_offset,
                    base_y + i * 15,
                    x_offset + bar_len,
                    base_y + i * 15 + 10,
                    fill="blue"
                )

                canvas.create_text(
                    x_offset - 15,
                    base_y + i * 15 + 5,
                    text=str(i),
                    font=("Arial", 8)
                )
