from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-1m.zip"
    if not archive.exists():
        with (
            urllib.request.urlopen(MOVIELENS_1M_URL, timeout=60) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    return destination / "ml-1m"
