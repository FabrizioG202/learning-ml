# %%
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from common.chunk_dataset import ChunkDataset
from vendor import bonito_download
import torch.nn as nn

# %%
DATA_PATH = bonito_download.download_file(
    Path("../.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)


# %%
dataset = ChunkDataset(DATA_PATH)

# %%
first_chunk, first_reference, first_reference_length = dataset[0]

model = nn.Sequential(
    nn.Conv1d(
        in_channels=1,

    )
)