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


DataPoint = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Basecaller:
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

    @staticmethod
    def decode_sequence(seq: torch.Tensor) -> str:
        """
        Decodes the given sequence into nucleotides.
        """
        return "".join("ACGT"[i - 1] if i != 0 else "-" for i in seq)

    def train(self, epochs: int) -> np.ndarray:
        """
        Trains the model, returns a 2D array with size (epochs, 2)
        where first row is train loss and second row is validation loss
        """
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0

            for x, y, target_lengths in train_loader:
                optimizer.zero_grad()

                x = x.to("cuda")
                y = y.to("cuda")

                # Forward pass
                y_ = model(x)

                # CTC expects inputs in the shape (length, batches, # of classes)
                log_probs = y_.permute(2, 0, 1).log_softmax(dim=2)

                # input lengths are the number of timesteps per each sample, which
                # is the same for all the samples
                input_lengths = torch.full(
                    (x.shape[0],),
                    y_.shape[2],
                    dtype=torch.long,
                )

                loss = criterion(log_probs, y, input_lengths, target_lengths)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.shape[0]

            train_losses.append(total_loss / len(train_loader.dataset))  # ty: ignore[invalid-argument-type]

            model.eval()
            total_loss = 0

            with torch.inference_mode():
                if len(val_loader.dataset) == 0:  # ty: ignore[invalid-argument-type]
                    continue

                for x, y, target_lengths in val_loader:
                    x = x.to("cuda")
                    y = y.to("cuda")

                    # Forward pass
                    y_ = model(x)

                    # CTC expects inputs in the shape (length, batches, # of classes)
                    log_probs = y_.permute(2, 0, 1).log_softmax(dim=2)

                    # input lengths are the number of timesteps per each sample, which
                    # is the same for all the samples
                    input_lengths = torch.full(
                        (x.shape[0],),
                        y_.shape[2],
                        dtype=torch.long,
                    )

                    loss = criterion(log_probs, y, input_lengths, target_lengths)

                    total_loss += loss.item() * y.shape[0]

            val_losses.append(total_loss / len(val_loader.dataset))  # ty: ignore[invalid-argument-type]

            if early_stopping.step(val_losses[-1], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return np.array([train_losses, val_losses])

    # @property
    # def device(self) -> str
    # return model.device.type

    def raw_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the sequence given the input x.
        Returns the largest logit for each timestep,
        which is a tensor of shape (batch_size, num_classes, timesteps)
        """
        assert x.ndim == 3, "Input must be a 3D tensor"

        model.eval()

        with torch.inference_mode():
            x = x.to("cuda")

            # (Forward pass)
            y_ = model(x)

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

    def compute_test_cer(
        self,
    ) -> float:
        """
        Computes the CER (Character Error Rate) on the test set and returns it as a float
        """

        self.model.eval()

        cers: list[float] = []

        with torch.inference_mode():
            for x, y, target_lengths in test_loader:
                x = x.to("cuda")
                y = y.to("cuda")

                y_ = model(x)
                batched_predictions = y_.argmax(1).cpu()

                for i in range(len(x)):
                    prediction = batched_predictions[i]

                    # collapse consequetive predictions
                    # and remove blank tokens
                    prediction = torch.unique_consecutive(prediction)
                    prediction = prediction[prediction != 0]

                    # decode the sequence and the expected sequence
                    decoded_predictions = decode_sequence(prediction)
                    decoded_expected = decode_sequence(

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


# %%
train_loader, val_loader, test_loader = (
    torch.utils.data.DataLoader(
        d,
        batch_size=256,
        shuffle=shuffle,
        pin_memory=True,
    )
    for (d, shuffle) in zip(
        random_split(
            Subset(ChunkDataset.full_from_path(DATA_PATH), range(100_000)),
            [0.7, 0.15, 0.15],
            generator=torch.Generator().manual_seed(0),
        ),
        [True, False, False],
    )
)

print(
    f"Dataset sizes train = {len(train_loader.dataset)}, "  # ty: ignore[invalid-argument-type]
    f"val = {len(val_loader.dataset)}, "  # ty: ignore[invalid-argument-type]
    f"test = {len(test_loader.dataset)}"  # ty: ignore[invalid-argument-type]
)


model = nn.Sequential(
    nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
    nn.ReLU(),
    nn.Conv1d(32, 128, kernel_size=15, stride=2, padding=7),
    nn.ReLU(),
    nn.Conv1d(128, 64, kernel_size=15, padding=7),
    nn.ReLU(),
    nn.Conv1d(64, 5, kernel_size=1),
).to("cuda")

N_EPOCHS = 100

criterion = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
early_stopping = EarlyStopping(patience=10)


# %%
# save the train and val loss as a 2d numpy array
np.savez(
    Path(DATA_PATH).parent / "losses.npz",
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
)

# %%
early_stopping.restore(model)

# %%
# load them back up
with np.load(Path(DATA_PATH).parent / "losses.npz") as data:
    train_losses = data["train_losses"].tolist()
    val_losses = data["val_losses"].tolist()

pio.renderers.default = "browser"

fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name="train loss"))
fig.add_trace(go.Scatter(y=val_losses, name="val loss"))
fig.show()


# %%
def decode_sequence(seq: torch.Tensor) -> str:
    return "".join("ACGT"[i - 1] if i != 0 else "-" for i in seq)


model.eval()

total_edit_distance = 0
sample_count = 0

with torch.inference_mode():
    x, y, target_lengths = next(iter(test_loader))

    x = x.to("cuda")
    y = y.to("cuda")

    y_ = model(x)
    predictions = y_.argmax(1).cpu()

    for i in range(len(x)):
        pred = predictions[i]

        # print("Raw predictions:")
        # print(decode_sequence(pred[:150]))

        counts = torch.bincount(pred, minlength=5)

        for symbol, count in zip("-ACGT", counts.tolist()):
            print(f"{symbol}: {count / len(pred):.2%}")

        # collapse consequetive predictions
        pred = torch.unique_consecutive(pred)
        pred = pred[pred != 0]  # remove blank tokens

        # decode the sequence
        pred_str = decode_sequence(pred)

        ref_ids = y[i, 0 : target_lengths[i]].tolist()
        expected = decode_sequence(torch.tensor(ref_ids))

        print(f"Expected: {expected}")
        print(f"Predicted: {pred_str}")

        edit_distance = Levenshtein.distance(expected, pred_str)
        print(f"Edit distance: {edit_distance}")
        total_edit_distance += edit_distance
        sample_count += 1
        break

print(f"Average edit distance: {total_edit_distance / sample_count:.2f}")
