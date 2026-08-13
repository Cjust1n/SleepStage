"""Streaming dataset builder to avoid OOM.

This module provides an alternative to DatasetBuilder.build_dataset/save
that processes the raw dataset incrementally and writes intermediates to disk.

Final outputs are written to:
  processed/X.npy
  processed/y.npy
  processed/metadata.csv

Strategy:
1) Iterate nights (no need to keep all NightData in RAM).
2) For each night: build_night -> X_n, y_n, meta_n.
3) Write each night chunk to disk as .npy files.
4) In a second pass, merge chunks into final arrays using np.memmap.

This avoids accumulating all X/y in Python lists, which is the main RAM issue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from numpy.lib.format import open_memmap

from Dataset.dataset_builder import DatasetBuilder
from Dataset.feature_order import FEATURE_ORDER



@dataclass
class ChunkRef:
    x_path: Path
    y_path: Path
    meta_path: Path
    n_rows: int


class StreamingDatasetBuilder:
    """Wrapper around existing DatasetBuilder.build_night()."""

    def __init__(self):
        self.builder = DatasetBuilder()

    def build_and_save_batched(
        self,
        nights: Iterable,
        output_dir: Path,
        *,
        batch_size: int = 10,
        intermediate_dir_name: str = "batches",
        # If provided, the built dataset will only contain these feature columns,
        # written into feature_names.json in the same order.
        selected_features_path: str | Path | None = None,
        # If provided (and selected_features_path is None), use this list.
        selected_features: List[str] | None = None,
    ):

        """Process nights in batches.

        - Builds each night and writes intermediate chunk files.
        - Merge final arrays using memmap.

        Returns
        -------
        dict with metadata about written dataset.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        intermediate_dir = output_dir / intermediate_dir_name
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        chunks: List[ChunkRef] = []

        batch_idx = 0
        night_buf = []

        def flush_batch(buf, batch_idx_: int):
            nonlocal chunks
            if not buf:
                return

            for chunk_in_batch_idx, night in enumerate(buf):
                print(f"[batch {batch_idx_:06d}] night {chunk_in_batch_idx+1}/{len(buf)}: {night.subject} Night {night.night}")

                X_n, y_n, meta_n = self.builder.build_night(night)
                if X_n.size == 0:
                    continue

                # Save as chunk files (small, per-night)
                chunk_id = f"b{batch_idx_:06d}_n{chunk_in_batch_idx:06d}"

                x_path = intermediate_dir / f"X_{chunk_id}.npy"
                y_path = intermediate_dir / f"y_{chunk_id}.npy"
                meta_path = intermediate_dir / f"metadata_{chunk_id}.csv"

                np.save(x_path, X_n.astype(np.float32, copy=False))
                np.save(y_path, y_n.astype(np.int32, copy=False))

                # meta_n dari DatasetBuilder.build_night() adalah list[dict], jadi convert dulu.
                pd.DataFrame(meta_n).to_csv(meta_path, index=False)


                chunks.append(
                    ChunkRef(
                        x_path=x_path,
                        y_path=y_path,
                        meta_path=meta_path,
                        n_rows=int(y_n.shape[0]),
                    )
                )

        for night in nights:
            night_buf.append(night)
            if len(night_buf) >= batch_size:
                flush_batch(night_buf, batch_idx)
                batch_idx += 1
                night_buf = []

        flush_batch(night_buf, batch_idx)

        if not chunks:
            # no data
            raise RuntimeError("No samples produced; check dataset paths/labels.")

        # Resolve selected features (optional)
        if selected_features is None and selected_features_path is not None:
            p = Path(selected_features_path)
            selected_features = [
                line.strip() for line in p.read_text().splitlines() if line.strip()
            ]

        if selected_features is not None:
            selected_features = list(selected_features)
            # Map full FEATURE_ORDER -> indices
            from Dataset.feature_order import FEATURE_ORDER

            name_to_idx = {n: i for i, n in enumerate(FEATURE_ORDER)}
            missing = [f for f in selected_features if f not in name_to_idx]
            if missing:
                raise KeyError(
                    "selected_features contains feature(s) not present in FEATURE_ORDER: "
                    + ", ".join(missing)
                )
            feature_idxs = [name_to_idx[f] for f in selected_features]
            feature_dim = int(len(feature_idxs))
        else:
            feature_dim = int(np.load(chunks[0].x_path, mmap_mode="r").shape[1])
            feature_idxs = None


        total_rows = sum(c.n_rows for c in chunks)

        # Pre-allocate final arrays via memmap
        x_final_path = output_dir / "X.npy"
        y_final_path = output_dir / "y.npy"
        meta_final_path = output_dir / "metadata.csv"

        # Create .npy files with valid NumPy headers, backed by memory mapping
        x_mm = open_memmap(
            x_final_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, feature_dim),
        )
        y_mm = open_memmap(
            y_final_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_rows,),
        )


        # Merge metadata incrementally
        meta_chunks: List[pd.DataFrame] = []
        # NOTE: metadata is much smaller than X/y; still, we avoid building
        # one huge Python list of X/y.

        offset = 0
        for i, c in enumerate(chunks):
            X_c = np.load(c.x_path)
            y_c = np.load(c.y_path)

            if feature_idxs is not None:
                X_c = X_c[:, feature_idxs]

            n = X_c.shape[0]
            x_mm[offset : offset + n, :] = X_c
            y_mm[offset : offset + n] = y_c
            offset += n


            meta_chunks.append(pd.read_csv(c.meta_path))

            print(f"Merging chunk {i+1}/{len(chunks)} rows={n} total_offset={offset}")

        x_mm.flush()
        y_mm.flush()

        # Write metadata
        metadata_all = pd.concat(meta_chunks, ignore_index=True)
        metadata_all.to_csv(meta_final_path, index=False)

        # Write feature_names.json
        feature_names_to_write = selected_features if selected_features is not None else FEATURE_ORDER
        with open(output_dir / "feature_names.json", "w") as f:
            json.dump(feature_names_to_write, f, indent=4)


        return {
            "X_shape": (total_rows, feature_dim),
            "y_shape": (total_rows,),
            "rows": total_rows,
            "chunks": len(chunks),
        }

