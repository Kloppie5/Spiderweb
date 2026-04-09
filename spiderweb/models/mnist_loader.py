
import struct

def load_images(path):
    with open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()

    images = []
    for i in range(num):
        start = i * rows * cols
        img = list(data[start:start + rows * cols])

        # normalized to [0,1]
        img = [x / 255.0 for x in img]

        images.append(img)

    return images

def load_labels(path):
    with open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = list(f.read())

    return labels
