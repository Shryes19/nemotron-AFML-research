"""Download only what we need from the Word2World dataset (X1AOX1A/LLMasWorldModels):
ALFWorld game files (their put->move grammar) and the four train/test split files.
Usage: python scripts/fetch_data.py
"""
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "X1AOX1A/LLMasWorldModels"
SPLITS = {
    "train/alfworld_train.json": "alfworld_train.json",
    "eval/alfworld_test.json": "alfworld_test.json",
    "train/sciworld_train.json": "sciworld_train.json",
    "eval/sciworld_test.json": "sciworld_test.json",
}


def main():
    repo_root = Path(__file__).resolve().parent.parent
    split_dir = repo_root / "splits"
    split_dir.mkdir(exist_ok=True)
    for remote, local in SPLITS.items():
        path = hf_hub_download(REPO_ID, remote, repo_type="dataset")
        shutil.copy(path, split_dir / local)
        print("split:", split_dir / local)

    alf_dir = Path(os.path.expanduser("~/.cache/alfworld"))
    if (alf_dir / "json_2.1.1").exists():
        print("ALFWorld data already at", alf_dir)
        return
    zip_path = hf_hub_download(REPO_ID, "alfworld.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(os.path.expanduser("~/.cache"))
    print("ALFWorld data extracted to", alf_dir)


if __name__ == "__main__":
    main()
