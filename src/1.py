# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

# %%
train_loader, val_loader, test_loader = (
    torch.utils.data.DataLoader(
        dataset=d,
        batch_size=1024,
        shuffle=True,
    )
    for d in [
        *torch.utils.data.random_split(
            MNIST(
                download=True,
                root="../.data/",
                train=True,
                transform=transforms.ToTensor(),
            ),
            [0.95, 0.05],
            generator=torch.Generator().manual_seed(42),
        ),
        MNIST(
            download=True,
            root="../.data/",
            train=False,
            transform=transforms.ToTensor(),
        ),
    ]
)


# %%

model = nn.Sequential(
    #
    # first conv layer
    nn.LazyConv2d(16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    #
    # second conv layer
    nn.LazyConv2d(32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    #
    # dense
    nn.Flatten(),
    nn.LazyLinear(64),
    nn.ReLU(),
    #
    # Out Layer
    nn.LazyLinear(10),  # since 10 classes
).to("cuda")

# lazy layer initializations
with torch.no_grad():
    _ = model(torch.zeros(1, 28, 28, device="cuda").unsqueeze(0))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses: list[float] = []
val_losses: list[float] = []
val_accuracies: list[float] = []

N_EPOCHS = 100

for i in tqdm(range(N_EPOCHS)):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        x_batch = x_batch.to("cuda")
        y_batch = y_batch.to("cuda")

        y_ = model(x_batch)
        loss = criterion(y_, y_batch)

        total_loss += loss.item() * x_batch.size(0)

        loss.backward()
        optimizer.step()

    train_losses.append(total_loss / len(train_loader))

    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.inference_mode():
        for x_batch, y_batch in val_loader:
            logits = model(x_batch.to("cuda"))
            loss = criterion(logits, y_batch.to("cuda"))

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (logits.argmax(1) == y_batch.to("cuda")).sum().item()

    val_losses.append(total_loss / len(val_loader))
    val_accuracies.append(total_correct / len(val_loader))


# %%
pio.renderers.default = "browser"

fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name="train loss"))
fig.add_trace(go.Scatter(y=val_losses, name="val loss"))
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(y=val_accuracies, name="val accuracy"))
fig.show()
