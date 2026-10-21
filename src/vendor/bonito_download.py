
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests
from tqdm import tqdm


class BaseDownloader(ABC):
    @abstractmethod
    def download(self, fname: str):
        pass


class Printer(BaseDownloader):
    def __init__(self):
        print("[available models]", file=sys.stderr)

    def download(self, fname: str):
        print(f" - {fname}", file=sys.stderr)


class Downloader(BaseDownloader):
    """
    Small class for downloading models and training assets.
    """

    __url__ = "https://cdn.oxfordnanoportal.com/software/analysis/bonito"

    def __init__(self, out_dir: Path, force: bool = False):
        print(f"[Downloading to {out_dir}]", file=sys.stderr)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.path = out_dir
        self.force = force

    def download(self, fname: str):
        url = f"{self.__url__}/{fname}.zip"
        fpath = self.path / f"{fname}"
        fpath_zip = self.path / f"{fname}.zip"

        if fpath.exists():
            if self.force:
                fpath.unlink() if fpath.is_file() else shutil.rmtree(fpath)
            else:
                print(f" - Skipping: {fname}", file=sys.stderr)
                return fpath

        # create the requests for the file
        req = requests.get(url, stream=True)
        total = int(req.headers.get("content-length", 0))

        # download the file content
        with (
            tqdm(
                total=total,
                unit="iB",
                ascii=True,
                ncols=100,
                unit_scale=True,
                leave=False,
            ) as t,
            fpath_zip.open("wb") as f,
        ):
            for data in req.iter_content(1024):
                if b"Error" in data:
                    raise FileNotFoundError(
                        f" - Failed to download: {fname}\n{data.decode()}"
                    )
                f.write(data)
                t.update(len(data))
        self._unzip(fpath_zip)
        print(f" - Downloaded: {fname}", file=sys.stderr)
        return fpath

    def _unzip(self, fpath):
        unzip_path = fpath.parent.with_suffix("")
        with ZipFile(fpath, "r") as zfile:
            zfile.extractall(path=unzip_path)
        fpath.unlink()
        return unzip_path

def download_file(
    out_dir: Path, file: str, show: bool = False, force: bool = False
) -> Path | None:
    dl: BaseDownloader = Printer() if show else Downloader(out_dir, force)
    try:
        dl.download(file)
        return out_dir / file

    except FileNotFoundError:
        print(f" - Failed to download: {file}")


ALL_TRAINING_FILES = [
    "example_data_dna_r9.4.1_v0",
    "example_data_dna_r10.4.1_v0",
    "example_data_rna004_v0",
]