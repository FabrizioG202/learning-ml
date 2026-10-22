# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from vendor import bonito_download
from vendor.early_stopping import EarlyStopping

# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)


# %%
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


class BasecallerHarness:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    model: torch.nn.Module
    criterion: torch.nn.Module
    early_stopping: EarlyStopping

    def __init__(
        self,
        *,
        split_ratios: list[float],
        dataset_size: int,
        model: nn.Module,
    ):
        self.model = model
        self.criterion = nn.CTCLoss(blank=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        self.early_stopping = EarlyStopping(patience=10)

        self.train_loader, self.val_loader, self.test_loader = (
            torch.utils.data.DataLoader(
                d,
                batch_size=256,
                shuffle=shuffle,
                pin_memory=True,
            )
            for (d, shuffle) in zip(
                random_split(
                    Subset(ChunkDataset.full_from_path(DATA_PATH), range(dataset_size)),
                    split_ratios,
                    generator=torch.Generator().manual_seed(0),
                ),
                [True, False, False],
            )
        )

    def log_dataset_size(self) -> None:
        """
        Logs the size of the dataset to the console.
        """
        print(
            f"Dataset sizes train = {len(self.train_loader.dataset)}, "  # ty: ignore[invalid-argument-type]
            f"val = {len(self.val_loader.dataset)}, "  # ty: ignore[invalid-argument-type]
            f"test = {len(self.test_loader.dataset)}"  # ty: ignore[invalid-argument-type]
        )

    @staticmethod
    def decode_sequence(seq: torch.Tensor) -> str:
        """
        Decodes the given sequence into nucleotides.
        """
        return "".join("ACGT"[i - 1] if i != 0 else "-" for i in seq)

    def train(self, epochs: int) -> Losses:
        """
        Trains the model, returns a 2D array with size (epochs, 2)
        where first row is train loss and second row is validation loss
        """
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(epochs)):
            ### TRAINING
            self.model.train()
            total_loss = 0

            for x, y, target_lengths in self.train_loader:
                self.optimizer.zero_grad()

                x = x.to("cuda")
                y = y.to("cuda")

                # Forward pass
                y_ = self.model(x)

                # CTC expects inputs in the shape (length, batches, # of classes)
                log_probs = y_.permute(2, 0, 1).log_softmax(dim=2)

                # input lengths are the number of timesteps per each sample, which
                # is the same for all the samples
                input_lengths = torch.full(
                    (x.shape[0],),
                    y_.shape[2],
                    dtype=torch.long,
                )

                loss = self.criterion(log_probs, y, input_lengths, target_lengths)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * y.shape[0]

            train_losses.append(total_loss / len(self.train_loader.dataset))  # ty: ignore[invalid-argument-type]

            ### EVALUATION
            self.model.eval()
            total_loss = 0

            with torch.inference_mode():
                if len(self.val_loader.dataset) == 0:  # ty: ignore[invalid-argument-type]
                    continue

                for x, y, target_lengths in self.val_loader:
                    x = x.to("cuda")
                    y = y.to("cuda")

                    # Forward pass
                    y_ = self.model(x)

                    # CTC expects inputs in the shape (length, batches, # of classes)
                    log_probs = y_.permute(2, 0, 1).log_softmax(dim=2)

                    # input lengths are the number of timesteps per each sample, which
                    # is the same for all the samples
                    input_lengths = torch.full(
                        (x.shape[0],),
                        y_.shape[2],
                        dtype=torch.long,
                    )

                    loss = self.criterion(log_probs, y, input_lengths, target_lengths)

                    total_loss += loss.item() * y.shape[0]

            val_losses.append(total_loss / len(self.val_loader.dataset))  # ty: ignore[invalid-argument-type]

            if self.early_stopping.step(val_losses[-1], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return Losses(np.array([train_losses, val_losses]))

    def raw_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the sequence given the input x.
        Returns the largest logit for each timestep,
        which is a tensor of shape (batch_size, num_classes, timesteps)
        """
        assert x.ndim == 3, "Input must be a 3D tensor"

        self.model.eval()

        with torch.inference_mode():
            x = x.to("cuda")

            # (Forward pass)
            y_ = self.model(x)

            # Get the largest logit.
            predictions = y_.argmax(1).cpu()

            return predictions

    def predict(self, x: torch.Tensor) -> list[str]:
        """
        Predicts the sequence given the input x.
        Returns a list of strings, one for each sample in the batch.
        """
        predictions = self.raw_predict(x)

        decoded_predictions = []

        for i in range(len(x)):
            prediction = predictions[i]

            # collapse consequetive predictions
            # and remove blank tokens
            prediction = torch.unique_consecutive(prediction)
            prediction = prediction[prediction != 0]

            # decode the sequence
            decoded_predictions.append(self.decode_sequence(prediction))

        return decoded_predictions

    def compute_test_cer(self) -> float:
        """
        Computes the CER (Character Error Rate) on the test set and returns it as a float
        """

        self.model.eval()

        cers: list[float] = []

        with torch.inference_mode():
            for x, y, target_lengths in self.test_loader:
                x = x.to("cuda")
                y = y.to("cuda")

                y_ = self.model(x)
                batched_predictions = y_.argmax(1).cpu()

                for i in range(len(x)):
                    prediction = batched_predictions[i]

                    # collapse consequetive predictions
                    # and remove blank tokens
                    prediction = torch.unique_consecutive(prediction)
                    prediction = prediction[prediction != 0]

                    # decode the sequence and the expected sequence
                    decoded_predictions = BasecallerHarness.decode_sequence(prediction)
                    decoded_expected = BasecallerHarness.decode_sequence(
                        # TODO: Not sure if torch.tensor and tolist are cancelling
                        # themselves out...
                        torch.tensor(
                            y[i, 0 : target_lengths[i]].tolist(),
                        )
                    )

                    # compute the edit distance between the two
                    # sequences. TODO: Not sure if this is the best way
                    # we might use a better distance metric in the future.
                    edit_distance = Levenshtein.distance(
                        decoded_expected, decoded_predictions
                    )

                    # cer is just the edit_distance normalized
                    # by the length of the expected sequence
                    cers.append(edit_distance / len(decoded_expected))

        return np.mean(cers)


caller = BasecallerHarness(
    model=nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
        nn.ReLU(),
        nn.Conv1d(32, 128, kernel_size=15, stride=2, padding=7),
        nn.ReLU(),
        nn.Conv1d(128, 64, kernel_size=15, padding=7),
        nn.ReLU(),
        nn.Conv1d(64, 5, kernel_size=1),
    ).to("cuda"),
    split_ratios=[0.8, 0.15, 0.05],
    dataset_size=1_000,
)

caller.log_dataset_size()
losses = caller.train(epochs=10)

# %%

pio.renderers.default = "browser"
losses.plot().show()
