from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

import numpy as np


class JSC(Dataset):
    """
    Jet Substructure Classification (JSC), OpenML id=42468 (hls4ml_lhc_jets_hlf).
    Tabular dataset: 16 float features, 5 classes.

    CIFAR-10-like API:
      - root, train, transform, target_transform, download
      - cached processed .pt for fast reload
    """

    openml_id = 42468

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Tensor], Tensor]] = None,
        download: bool = False,
        split_seed: int = 1337,
        train_fraction: float = 0.8,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.raw_dir = self.root / "JSC" / "raw"
        self.proc_dir = self.root / "JSC" / "processed"
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        # cache file keyed by seed & fraction so you can reproduce splits
        split_tag = f"seed{split_seed}_train{train_fraction:.2f}"
        self.cache_path = self.proc_dir / f"jsc_{split_tag}.pt"

        if download:
            self._download_and_process(split_seed=split_seed, train_fraction=train_fraction)

        if not self.cache_path.exists():
            raise RuntimeError(
                f"Processed file not found: {self.cache_path}\n"
                "Pass download=True to create it."
            )

        obj = torch.load(self.cache_path, map_location="cpu")
        if train:
            self.X: Tensor = obj["X_train"].to(dtype)
            self.y: Tensor = obj["y_train"]
        else:
            self.X: Tensor = obj["X_test"].to(dtype)
            self.y: Tensor = obj["y_test"]

        if self.normalize:
            mean = self.X.mean(dim=0, keepdim=True)
            std = self.X.std(dim=0, keepdim=True)
            self.X = (self.X - mean) / std
        # Optional metadata
        self.feature_names = obj.get("feature_names", None)
        self.class_names = obj.get("class_names", None)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def _download_and_process(self, split_seed: int, train_fraction: float) -> None:
        # If cache exists, don't redo work.
        if self.cache_path.exists():
            return

        # Use scikit-learn OpenML loader and force ARFF parsing to avoid parquet/pyarrow issues.
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # fetch_openml returns (X, y) where X is array-like and y is labels
        # parser="liac-arff" avoids parquet; good fallback when pyarrow breaks.
        bunch = fetch_openml(
            data_id=self.openml_id,
            as_frame=True,          # DataFrame -> easier to handle mixed types
            parser="liac-arff",     # important: avoid parquet/arrow path
        )

        X_df = bunch.data
        y_raw = bunch.target

        # Convert features to float32
        X = X_df.to_numpy(dtype=np.float32, copy=False)

        # Encode labels to int64 [0..K-1]
        le = LabelEncoder()
        y = le.fit_transform(np.asarray(y_raw)).astype(np.int64)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=train_fraction,
            random_state=split_seed,
            stratify=y
        )

        # Save as torch tensors
        obj = {
            "X_train": torch.from_numpy(X_train),
            "y_train": torch.from_numpy(y_train),
            "X_test": torch.from_numpy(X_test),
            "y_test": torch.from_numpy(y_test),
            "feature_names": list(X_df.columns),
            "class_names": list(le.classes_),
            "openml_id": self.openml_id,
            "split_seed": split_seed,
            "train_fraction": train_fraction,
        }
        torch.save(obj, self.cache_path)
