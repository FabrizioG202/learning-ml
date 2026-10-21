# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from vendor import bonito_download

# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)

# %%
train_loader, test_loader, val_loader = (
    torch.utils.data.DataLoader(
        d,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
    )
    for d in ChunkDataset.full_from_path(DATA_PATH)
    .take(10_000)
    .tri_split(test_ratio=0.75, val_ratio=0.05, seed=42)
)

# for x_, y_, z in (train_loader):
#     print(x_.shape)
#     break

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

