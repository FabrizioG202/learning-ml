from pathlib import Path

import numpy as np
import plotly.graph_objects as go


class Losses:
    _losses: np.ndarray

    def __init__(self, losses: np.ndarray):
        self._losses = losses

    @property
    def train(self) -> list[float]:
        return self._losses[0].tolist()

    @property
    def validation(self) -> list[float]:
        return self._losses[1].tolist()

    def save(self, path: Path) -> None:
        np.savez(path, train_losses=self.train, val_losses=self.validation)

    @staticmethod
    def load(path: Path) -> "Losses":
        with np.load(path) as data:
            train_losses = data["train_losses"].tolist()
            val_losses = data["val_losses"].tolist()

        return Losses(np.array([train_losses, val_losses]))

    def plot(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.train, name="Train Loss"))
        fig.add_trace(go.Scatter(y=self.validation, name="Validation Loss"))
        return fig
