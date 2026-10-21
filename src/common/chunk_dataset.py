from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    # Chunk data, shape: (num_chunks, 5000 (chunk length))
    chunks: np.ndarray

    # Reference data, shape: (num_chunks, 500 (sequence length))
    # Sequences are padded with 0 to length of 500.
    references: np.ndarray

    # Actual lengths of reference sequences, shape: (num_chunks,)
    reference_lengths: np.ndarray

    def __init__(self, data_dir: Path):
        self.chunks, self.references, self.reference_lengths = (
            np.load(data_dir / f, mmap_mode="r")
            for f in ["chunks.npy", "references.npy", "reference_lengths.npy"]
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.chunks[index], copy=True)).float(),
            torch.from_numpy(np.array(self.references[index], copy=True)).long(),
            torch.tensor(int(self.reference_lengths[index])).long(),
        )

    def id_to_symbol(self, id: int) -> str:
        return "ACGT"[id - 1] if id != 0 else "-"

    def take(self, n: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [self[i] for i in range(n)]
