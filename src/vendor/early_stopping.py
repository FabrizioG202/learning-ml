import copy
import math

from torch import nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state = None
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if not math.isfinite(val_loss):
            return True

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
