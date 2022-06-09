from cProfile import label
import numpy as np

class CrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor_e = prediction_tensor + np.finfo(np.float64).eps
        prediction_tensor_mask = self.prediction_tensor_e[label_tensor > 0]
        return -np.sum(np.log(prediction_tensor_mask))

    def backward(self, label_tensor):
        return -(label_tensor / self.prediction_tensor_e)
