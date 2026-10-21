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

    def __init__(
        self, chunks: np.ndarray, references: np.ndarray, reference_lengths: np.ndarray
    ):
        self.chunks = chunks
        self.references = references
        self.reference_lengths = reference_lengths

    @staticmethod
    def full_from_path(data_dir: Path) -> "ChunkDataset":

        return ChunkDataset(
            *(
                np.load(data_dir / f, mmap_mode="r")
                for f in ["chunks.npy", "references.npy", "reference_lengths.npy"]
            ),
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.chunks[index], copy=True))
            .unsqueeze(0)
            .float(),
            torch.from_numpy(np.array(self.references[index], copy=True)).long(),
            torch.tensor(int(self.reference_lengths[index])).long(),
        )

    def id_to_symbol(self, id: int) -> str:
        return "ACGT"[id - 1] if id != 0 else "-"

    def take(self, n: int) -> "ChunkDataset":
        return ChunkDataset(
            self.chunks[:n],
            self.references[:n],
            self.reference_lengths[:n],
        )

    ### NB: the datasets are shuffled
    def tri_split(
        self,
        *,
        test_ratio: float,
        val_ratio: float,
        seed: int | None = None,
    ) -> tuple["ChunkDataset", "ChunkDataset", "ChunkDataset"]:
        assert 0 <= test_ratio < 1, "test_ratio must be in [0, 1)"
        assert 0 <= val_ratio < 1, "val_ratio must be in [0, 1)"
        assert test_ratio + val_ratio < 1, "test_ratio + val_ratio must be less than 1"

        total = len(self)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)
        train_size = total - test_size - val_size

        indices = torch.randperm(total, generator=torch.manual_seed(seed) if seed is not None else None).numpy()

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        return (
            ChunkDataset(
                self.chunks[train_indices],
                self.references[train_indices],
                self.reference_lengths[train_indices],
            ),
            ChunkDataset(
                self.chunks[val_indices],
                self.references[val_indices],
                self.reference_lengths[val_indices],
            ),
            ChunkDataset(
                self.chunks[test_indices],
                self.references[test_indices],
                self.reference_lengths[test_indices],
            ),
        )
