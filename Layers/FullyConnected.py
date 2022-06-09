from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self.gradient_weights = None

        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self,input_tensor):
        _input_tensor = np.insert(input_tensor, input_tensor.shape[1] ,np.ones(input_tensor.shape[0]), axis=1)
        output=np.dot(_input_tensor,self.weights)
        self.input_tensor = _input_tensor
        return output

    def backward(self,error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(weight_tensor = self.weights, gradient_tensor = self.gradient_weights)
        return np.delete(np.dot(error_tensor,self.weights.T), (-1), axis=1)
