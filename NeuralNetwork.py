from copy import deepcopy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        return self.loss_layer.forward(self.test(self.input_tensor), self.label_tensor)

    def backward(self):
        output = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            output = layer.backward(output)

        return output

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)

        self.layers.append(layer)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    def train(self, iterations) -> None:
        for i in range(0, iterations):
            self.loss.append(self.forward())
            self.backward()
