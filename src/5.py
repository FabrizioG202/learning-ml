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

train_losses = []
val_losses = []

for epoch in tqdm(range(N_EPOCHS)):
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
