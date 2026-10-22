# %%
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.io as pio
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from common.losses import Losses
from vendor import bonito_download
from vendor.early_stopping import EarlyStopping

# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)
type DataPoint = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]  # (x, y, target_lengths)


# %%
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
        batch_size: int,
        checkpoint_save_path: str,
        dataset_sample_seed: int | None = None,
    ):
        self.model = model
        self.criterion = nn.CTCLoss(blank=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        self.early_stopping = EarlyStopping(
            patience=10,
            path=checkpoint_save_path,
            verbose=False,
        )

        dataset = ChunkDataset.full_from_path(DATA_PATH)
        indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(dataset_sample_seed)
            if dataset_sample_seed is not None
            else None,
        )[:dataset_size].tolist()

        self.train_loader, self.val_loader, self.test_loader = (
            torch.utils.data.DataLoader(
                d,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
            )
            for (d, shuffle) in zip(
                random_split(
                    Subset(dataset, indices),
                    split_ratios,
                    generator=torch.Generator().manual_seed(dataset_sample_seed + 1)
                    if dataset_sample_seed is not None
                    else None,
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
        alphabet = "-ACGT"
        return "".join(alphabet[i] for i in seq.tolist())

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

            validation_loss = total_loss / len(self.val_loader.dataset)  # ty: ignore[invalid-argument-type]
            val_losses.append(validation_loss)

            if self.early_stopping(validation_loss, self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # restore the best model as per early stopping
        self.model.load_state_dict(torch.load(self.early_stopping.path))

        return Losses(np.array([train_losses, val_losses]))

    def raw_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the sequence given the input x.
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
        throws if the test set is empty
        """

        assert len(self.test_loader.dataset) > 0, "Test set is empty"  # ty: ignore[invalid-argument-type]

        # Put model in eval mode
        # TODO: Not sure if this is necessary since
        #       we're using torch.inference_mode() later.
        self.model.eval()

        # cers: list[float] = []
        total_distance = 0
        total_bases = 0

        with torch.inference_mode():
            for x, y, target_lengths in self.test_loader:
                predictions = self.predict(x)

                for predicted, target, length in zip(predictions, y, target_lengths):
                    # decode the sequence and the expected sequence
                    expected = BasecallerHarness.decode_sequence(
                        target[: int(length)].cpu(),
                    )

                    # compute the edit distance between the two
                    # sequences.
                    # TODO: Not sure if this is the best way
                    #       we might use a better distance metric in the future.
                    total_distance += Levenshtein.distance(expected, predicted)
                    total_bases += len(expected)

        return total_distance / total_bases

    def sample_random(
        self,
        seed: int | None,
        source: Literal[
            "train",
            "test",
            "val",
        ],
    ) -> DataPoint:
        """
        Samples a random data point from the specified source.
        """
        if source == "train":
            loader = self.train_loader
        elif source == "val":
            loader = self.val_loader
        elif source == "test":
            loader = self.test_loader

        if len(loader.dataset) == 0:  # ty: ignore[invalid-argument-type]
            raise ValueError(f"{source} set is empty")

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(loader.dataset))  # ty: ignore[invalid-argument-type]

        return loader.dataset[idx]

    def pair_predict(self, point: DataPoint) -> tuple[str, str]:
        """
        Predicts the sequence given the input x.
        Returns a tuple of (predicted_sequence, expected_sequence)
        """
        x, y, target_lengths = point

        predicted = self.predict(x.unsqueeze(0))[0]
        expected = BasecallerHarness.decode_sequence(
            y[: int(target_lengths)].cpu(),
        )

        return predicted, expected


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
    dataset_size=1000,
    batch_size=512,
    checkpoint_save_path="../.models/5.py",
)

caller.log_dataset_size()

# %%
losses = caller.train(epochs=1000)
# %%
random_sample = caller.sample_random(seed=42, source="test")
predicted, expected = caller.pair_predict(random_sample)

# %%
print(predicted, expected)

# %%
print(caller.compute_test_cer())


# %%
pio.renderers.default = "browser"

losses.plot().show()

# %%
