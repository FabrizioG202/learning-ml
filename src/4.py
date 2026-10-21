# %%
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import torch
from torch import nn
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from vendor import bonito_download

# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)

# %%
train_loader, val_loader, test_loader = (
    torch.utils.data.DataLoader(
        d,
        batch_size=8,
        shuffle=shuffle,
        pin_memory=True,
    )
    for (d, shuffle) in zip(
        random_split(
            Subset(ChunkDataset.full_from_path(DATA_PATH), range(1_000)),
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
    nn.Conv1d(
        in_channels=1,
        out_channels=16,
        kernel_size=5,
        stride=2,
        padding=2,
    ),
    nn.ReLU(),
    nn.Conv1d(16, 5, kernel_size=1),
).to("cuda")

N_EPOCHS = 10

criterion = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

for _ in tqdm(range(N_EPOCHS)):
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

# %%
pio.renderers.default = "browser"

fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name="train loss"))
fig.add_trace(go.Scatter(y=val_losses, name="val loss"))
fig.show()


# %%
def decode_sequence(seq: torch.Tensor) -> str:
    return "".join("ACGT"[i - 1] if i != 0 else "-" for i in seq)

model.eval()

with torch.inference_mode():
    x, y, target_lengths = next(iter(test_loader))

    x = x.to("cuda")
    y = y.to("cuda")

    y_ = model(x)
    predictions = y_.argmax(1).cpu()

    for i in range(len(x)):
        pred = predictions[i]

        counts = torch.bincount(pred, minlength=5)

        for symbol, count in zip("-ACGT", counts.tolist()):
            print(f"{symbol}: {count / len(pred):.2%}")

        # collapse consequetive predictions
        pred = torch.unique_consecutive(pred)
        pred = pred[pred != 0]  # remove blank tokens

        # decode the sequence
        pred_str = decode_sequence(pred)

        ref_ids = y[i, 0:target_lengths[i]].tolist()
        expected = decode_sequence(torch.tensor(ref_ids))

        print("Raw predictions:")
        print(decode_sequence(pred[:150]))
        print(f"Expected: {expected}")
        print(f"Predicted: {pred_str}")


