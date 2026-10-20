# %%
import re
from collections import Counter

import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

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

    def __len__(self) -> int:
        return len(self.mapping)


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
        batch_size=2048,
        shuffle=True,
        collate_fn=collate_fn,
    )
    for d in (split["train"], split["test"], dataset["test"])
]


# %%

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 64

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        embeddings = self.embedding(x)

        mask = (x != 0).unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        pooled = summed / counts

        return self.linear(pooled).squeeze(-1)


model = Model().to("cuda")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses: list[float] = []
val_losses: list[float] = []

N_EPOCH = 128
for i in tqdm(range(N_EPOCH)):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        x_batch = x_batch.to("cuda")
        y_batch = y_batch.to("cuda").float()

        y_ = model(x_batch)
        loss = criterion(y_, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.shape[0]

    train_losses.append(total_loss / len(train_loader.dataset))  # ty: ignore[invalid-argument-type]

    model.eval()
    total_loss = 0

    with torch.inference_mode():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to("cuda")
            y_batch = y_batch.to("cuda").float()

            y_ = model(x_batch)
            loss = criterion(y_, y_batch)

            total_loss += loss.item() * y_batch.shape[0]

    val_losses.append(total_loss / len(val_loader.dataset))  # ty: ignore[invalid-argument-type]


# %%
pio.renderers.default = "browser"

fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name="train loss"))
fig.add_trace(go.Scatter(y=val_losses, name="val loss"))
fig.show()

# %%
