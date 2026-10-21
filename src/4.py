# %%
from pathlib import Path

from vendor import bonito_download

# %%
DATA_PATH = bonito_download.download_file(
    Path("./.data"),
    bonito_download.ALL_TRAINING_FILES[1],
)
