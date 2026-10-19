# %%
import re
from collections import Counter

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# %%
dataset = load_dataset("stanfordnlp/imdb", cache_dir="../.data")


# %%
class Vocab:
    def __init__(self, mapping: dict[str, int]):
        self.mapping = mapping
        self.inverse_mapping = {v: k for k, v in mapping.items()}

    @staticmethod
    def __tokenize(text: str) -> list[str]:
        # NB: This function was taken from the internet.
        text = re.sub(r"<[^>]+>", " ", text.lower())
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    @staticmethod
    def create(sample_texts: list[str], size: int) -> "Vocab":
        counter = Counter()
        for example in sample_texts:
            counter.update(Vocab.__tokenize(example))

        mappings = {
            "<pad>": 0,  # padding
            "<unk>": 1,  # unknown tokens
        }

        for word, _ in counter.most_common(size - 2):
            mappings[word] = len(mappings)

        return Vocab(mappings)

    def get(self, word: str) -> int:
        return self.mapping.get(word, self.mapping["<unk>"])

    def tokenize_sentence(self, text: str) -> list[int]:
        return [self.get(t) for t in self.__tokenize(text)]


split = dataset["train"].train_test_split(test_size=0.05, seed=42)

vocab = Vocab.create(
    [text for text in split["train"]["text"]],
    size=20_000,
)


# %%
class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, vocab: Vocab, max_length: int):
        self.ds = dataset
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.ds[index]

        text: str = item["text"]
        label: int = item["label"]

        tokenized = self.vocab.tokenize_sentence(text)[: self.max_length]

        return (
            torch.tensor(tokenized or [1], dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_fn(batch):
    sequences, labels = zip(*batch)

    sequences = pad_sequence(
        sequences,  # ty: ignore[invalid-argument-type]
        batch_first=True,
        padding_value=0,
    )

    labels = torch.stack(labels)

    return sequences, labels


train_loader, val_loader, test_loader = [
    DataLoader(
        DatasetLoader(d, vocab, 256),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
    for d in (split["train"], split["test"], dataset["test"])
]


# %%

# %%
x, y = next(iter(train_loader))

print("Inputs:", x.shape, x.dtype)
print("Labels:", y.shape, y.dtype)
print("Vocabulary size:", len(vocab.mapping))
