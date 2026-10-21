# %%
from pathlib import Path

from vendor import bonito_download

# %%
RAW_DATA_DIR = Path("./raw-data")
bonito_download.download_files(RAW_DATA_DIR, bonito_download.ALL_TRAINING_FILES[1::])

DATA_PATH = RAW_DATA_DIR / " example_data_dna_r10.4.1_v0"

