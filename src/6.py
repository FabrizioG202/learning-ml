# %%
import dataclasses
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn
from torch.nn import LazyConv1d
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from common.losses import Losses
from vendor import bonito_download
from vendor.early_stopping import EarlyStopping

pio.renderers.default = "browser"

type DataPoint = tuple[
    torch.Tensor,  # x
    torch.Tensor,  # y
    torch.Tensor,  # target lengths
]


@dataclasses.dataclass
class ModelResult:
    model_name: str
    losses: Losses
    val_cer: float

@dataclasses.dataclass
class LRResult:
    model_name: str
    lr: float
    losses: Losses
    val_cer: float


# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)

# %%
#
# Preliminary data analysis
#
full_dataset = ChunkDataset.full_from_path(DATA_PATH)

# %%
#
# Sample chunk
#

first_chunk, _, _ = full_dataset[0]
fig = px.line(first_chunk.squeeze(0).numpy()[0:500])
fig.update_layout(
    showlegend=False, xaxis_title="Time", yaxis_title="Normalized Current"
)
fig.write_image("../docs/.images/current.svg")

# %%
#
# Length distribution
#


fig = px.histogram(full_dataset.reference_lengths, nbins=100).update_layout(
    xaxis_title="Reference Length",
    yaxis_title="Count",
    showlegend=False,
)

mean_length = np.mean(full_dataset.reference_lengths)
fig.add_vline(
    x=mean_length,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Mean: {mean_length:.2f}",
    annotation_position="top right",
)

fig.write_image("../docs/.images/reference_lengths.svg")

# %%
#
# Nucletide composition
#
counts = pd.DataFrame.from_dict(
    {
        "-ACGT"[k]: [v]
        for k, v in Counter(full_dataset.references.flatten().tolist()).items()
        if k != 0
    },
    orient="index",
    columns=["counts"],
)

px.histogram(counts, x=counts.index, y="counts").update_layout(
    xaxis_title="Nucleotide",
    yaxis_title="Count",
    showlegend=False,
).write_image("../docs/.images/nucleotide_composition.svg")


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
        lr: float = 0.0005,
    ):
        self.model = model
        self.criterion = nn.CTCLoss(blank=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    def all_data_points(self) -> Iterator[DataPoint]:
        yield from self.train_loader.dataset
        yield from self.val_loader.dataset
        yield from self.test_loader.dataset

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

        for epoch in tqdm(range(epochs), position=0):
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

                # gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(
                            f"Non-finite gradient in {name}, loss = {loss.item():.4f}"
                        )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, error_if_nonfinite=True
                )

                self.optimizer.step()

                total_loss += loss.item() * x.shape[0]

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

                    total_loss += loss.item() * x.shape[0]

            validation_loss = total_loss / len(self.val_loader.dataset)  # ty: ignore[invalid-argument-type]
            val_losses.append(validation_loss)

            self.early_stopping(validation_loss, self.model)

            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # restore the best model as per early stopping
        if (
            self.early_stopping.path is not None
            and Path(self.early_stopping.path).exists()
        ):
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

    def compute_val_cer(self) -> float:
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
            for x, y, target_lengths in self.val_loader:
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


class BasecallerModel(nn.Module):
    def __init__(
        self,
        first_cnn_layer: nn.LazyConv1d,
        second_cnn_layer: nn.LazyConv1d,
        third_cnn_layer: nn.LazyConv1d,
        lstm_hidden_size: int,
        lstm_num_layers: int,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # first convolutional layer
            first_cnn_layer,
            nn.LazyBatchNorm1d(),
            nn.SiLU(),  # swish activation function (like the one used in Bonito)
            second_cnn_layer,
            nn.LazyBatchNorm1d(),
            nn.SiLU(),
            third_cnn_layer,
            nn.LazyBatchNorm1d(),
            nn.SiLU(),
        )

        self.lstm = nn.LSTM(
            input_size=third_cnn_layer.out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0.0,
        )

        self.linear = nn.LazyLinear(5)

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        # (BATCH, 1, 5000) -> (8, featurs, time)
        x = self.cnn(x)

        # LSTM expects (B, time, features)
        x = x.transpose(1, 2)

        # LTSM layer
        x, _ = self.lstm(x)

        # (linear already operates on the last dimension (which is the feature))
        x = self.linear(x)

        # Fix the return value to match what we expcect.
        return x.transpose(1, 2)

# (the function below was generated by an LLM)
# TODO: This might actually be a constructor on the BasecallerModel class
def create_model(config: dict) -> BasecallerModel:
    return BasecallerModel(
        first_cnn_layer=nn.LazyConv1d(
            config["channels"][0],
            kernel_size=config["kernels"][0],
            stride=2,
            padding=config["kernels"][0] // 2,
        ),
        second_cnn_layer=nn.LazyConv1d(
            config["channels"][1],
            kernel_size=config["kernels"][1],
            stride=2,
            padding=config["kernels"][1] // 2,
        ),
        third_cnn_layer=nn.LazyConv1d(
            config["channels"][2],
            kernel_size=config["kernels"][2],
            stride=1,
            padding=config["kernels"][2] // 2,
        ),
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_num_layers=config["lstm_num_layers"],
    ).to("cuda")

# %%
#
# model selection
#

# We use factories so that we
# can recreate the best model.
MODEL_CONFIGS = {
    "small": {
        "channels": (16, 32, 64),
        "kernels": (15, 15, 15),
        "lstm_hidden_size": 128,
        "lstm_num_layers": 1,
    },
    "medium": {
        "channels": (32, 64, 128),
        "kernels": (15, 15, 15),
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
    },
    "large": {
        "channels": (32, 64, 128),
        "kernels": (15, 21, 31),
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
    },
}

SELECTION_DATASET_SIZE = 5_000

# (for reporoducibility purposes)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


results: list[ModelResult] = []

for name, config in MODEL_CONFIGS.items():
    model = create_model(config)

    model.eval()

    # (initialize lazy layers)
    with torch.no_grad():
        y_ = model(torch.zeros(1, 5000).unsqueeze(0).to("cuda"))

    harness = BasecallerHarness(
        model=model,
        split_ratios=[0.8, 0.05, 0.15],
        dataset_size=SELECTION_DATASET_SIZE,
        batch_size=64,
        checkpoint_save_path=f"../.models/6_{name}.pt",
        dataset_sample_seed=0,
    )

    losses = harness.train(epochs=100)

    results.append(ModelResult(name, losses, harness.compute_val_cer()))

best_model = min(results, key=lambda r: r.val_cer)
print(f"Best model: {best_model.model_name} with val_cer = {best_model.val_cer:.4f}")

# %%
#
# lr selection
#

lr_results : list[LRResult] = []

for lr in [0.001, 0.0005, 0.0001]:
    model = create_model(MODEL_CONFIGS[best_model.model_name])
    model.eval()

    # (initialize lazy layers)
    with torch.no_grad():
        y_ = model(torch.zeros(1, 5000).unsqueeze(0).to("cuda"))

    harness = BasecallerHarness(
        model=model,
        split_ratios=[0.8, 0.05, 0.15],
        dataset_size=SELECTION_DATASET_SIZE,
        batch_size=64,
        checkpoint_save_path=f"../.models/6_{name}_lr{lr}.pt",
        dataset_sample_seed=0,
        lr=lr,
    )

    losses = harness.train(epochs=100)

    lr_results.append(LRResult(best_model.model_name, lr, losses, harness.compute_val_cer()))

best_lr_model = min(lr_results, key=lambda r: r.val_cer)
print(f"Best model: {best_lr_model.model_name} with val_cer = {best_lr_model.val_cer:.4f} and lr = {best_lr_model.lr}")

# %%
random_sample = harness.sample_random(seed=42, source="test")
predicted, expected = harness.pair_predict(random_sample)

# %%
print(predicted, expected)

# %%
print(harness.compute_val_cer())


# %%

losses.plot().show()

# %%
