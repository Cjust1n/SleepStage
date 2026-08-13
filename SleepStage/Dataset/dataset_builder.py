"""
dataset_builder.py

Build final ML dataset.

Output:

X.npy
y.npy
metadata.csv
feature_names.json

Author : Justin
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter
from Preprocessing.cleaning import EpochCleaner
from Preprocessing.window_builder import WindowBuilder
from Preprocessing.sequence_builder import SequenceBuilder

from config import SEQUENCE_LENGTH

from FeatureExtraction import (
    HRVFeatureExtractor,
    MovementFeatureExtractor,
    TemporalFeatureExtractor,
)

from Dataset.feature_order import FEATURE_ORDER


class DatasetBuilder:

    def __init__(self):

        self.aligner = EpochAligner()

        self.segmenter = EpochSegmenter()

        self.cleaner = EpochCleaner()

        self.window_builder = WindowBuilder()

        self.hrv = HRVFeatureExtractor()

        self.motion = MovementFeatureExtractor()

        self.temporal = TemporalFeatureExtractor()

        self.sequence_builder = SequenceBuilder(
            sequence_length=SEQUENCE_LENGTH,
        )

    # ---------------------------------------------------------

    def build_night(self, night):

        motion, hr = self.aligner.align_night(night)

        motion_epochs, hr_epochs = self.segmenter.segment_night(
            motion,
            hr,
        )

        # Raw label mapping from ekspert annotations.
        # Raw: 0=wake, 1=N1, 2=N2, 3=N3, 4=REM
        labels = {i: int(label) for i, label in enumerate(night.expert_label)}

        # Remap labels into new 4-class scheme BEFORE saving into processed/y.npy.
        # New (contiguous 0..3):
        #   0 = wake
        #   1 = N1 & N2
        #   2 = N3
        #   3 = REM
        def _remap_label(lbl: int) -> int:
            # Raw label values observed/expected from dataset annotations:
            #   0=wake, 1=N1, 2=N2, 3=N3, 4=REM
            if lbl in (1, 2):
                return 1
            if lbl == 3:
                return 2
            if lbl == 4:
                return 3
            if lbl == 0:
                return 0


            # Some nights may contain an extra label id (e.g. 5) depending on
            # annotation source/preprocessing. Fail gracefully by dropping
            # these epochs rather than crashing the whole dataset build.
            #
            # NOTE: We map to None-like behavior by raising a sentinel error
            # that is caught below.
            raise ValueError(f"Unexpected label value: {lbl}")


        # Apply remap to the per-epoch labels; metadata remains aligned because
        # it is appended using the same loop over `epoch` keys.
        # If an unexpected raw label id appears, we DROP that epoch.
        cleaned_labels = {}
        for k, v in labels.items():
            try:
                cleaned_labels[k] = _remap_label(v)
            except ValueError:
                # drop this epoch
                continue
        labels = cleaned_labels



        motion_epochs, hr_epochs, labels = self.cleaner.clean(

            motion_epochs,
            hr_epochs,
            labels,

        )

        if hr_epochs:
            cleaned_hr = pd.concat(hr_epochs.values(), ignore_index=True)
            cleaned_hr.sort_values(by="Timestamp", inplace=True)
            cleaned_hr.reset_index(drop=True, inplace=True)
        else:
            cleaned_hr = pd.DataFrame(
                columns=["Timestamp", "HR", "epoch"]
            )

        X = []

        y = []

        metadata = []

        feature_dim = (
            len(FEATURE_ORDER)
        )

        total_epochs = len(night.expert_label)

        for epoch in sorted(labels.keys()):

            motion_feature = self.motion.extract(
                motion_epochs[epoch]
            )

            if epoch in hr_epochs and not hr_epochs[epoch].empty:
                current_timestamp = hr_epochs[epoch]["Timestamp"].max()
            else:
                current_timestamp = None

            hr_window = self.window_builder.get_hr_window(
                cleaned_hr,
                current_timestamp,
            )

            hrv_feature = self.hrv.extract(
                hr_window
            )

            temporal_feature = self.temporal.extract(

                epoch=epoch,

                total_epochs=total_epochs,

                rec_start=night.rec_start,

            )

            feature_dict = {}
            feature_dict.update(hrv_feature)
            feature_dict.update(motion_feature)
            feature_dict.update(temporal_feature)

            feature_vector = np.array(
                [feature_dict.get(name, 0.0) for name in FEATURE_ORDER],
                dtype=np.float32,
            ).reshape(1, -1)

            # Safety: drop any epoch that produces NaN/Inf features.
            # NaNs can cause training to collapse (loss becomes NaN).
            if not np.isfinite(feature_vector).all():
                continue

            X.append(feature_vector)

            y.append(labels[epoch])

            metadata.append({

                "Subject": night.subject,

                "Night": night.night,

                "Epoch": epoch,

            })

        if X:
            X_array = np.asarray(X, dtype=np.float32).reshape(-1, feature_dim)
            y_array = np.asarray(y, dtype=np.int32)
        else:
            X_array = np.empty((0, feature_dim), dtype=np.float32)
            y_array = np.empty((0,), dtype=np.int32)

        return (

            X_array,

            y_array,

            metadata,

        )

    # ---------------------------------------------------------

    def build_dataset(self, dataset):

        X_all = []

        y_all = []

        metadata_all = []

        for night in dataset:

            print(

                f"Processing {night.subject} Night {night.night}"

            )

            X, y, meta = self.build_night(night)

            if X.size > 0:
                X = np.asarray(X, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                elif X.ndim != 2:
                    X = X.reshape(-1, X.shape[-1])
                X_all.append(X)
                y_all.append(y)
                metadata_all.extend(meta)

        if not X_all:
            feature_dim = (
                len(FEATURE_ORDER)
            )
            return (
                np.empty((0, feature_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                pd.DataFrame(columns=["Subject", "Night", "Epoch"]),
            )

        X_all = np.concatenate(X_all, axis=0)

        y_all = np.concatenate(y_all, axis=0)

        metadata_all = pd.DataFrame(metadata_all)

        return X_all, y_all, metadata_all

    # ---------------------------------------------------------

    def build_sequences(self, dataset):

        X, y, metadata = self.build_dataset(dataset)

        return self.sequence_builder.build_sequences(
            X,
            y,
            metadata,
        )

    # ---------------------------------------------------------

    def save(

        self,

        output_folder,

        X,

        y,

        metadata,

    ):

        output_folder = Path(output_folder)

        output_folder.mkdir(

            parents=True,

            exist_ok=True,

        )

        np.save(

            output_folder / "X.npy",

            X,

        )

        np.save(

            output_folder / "y.npy",

            y,

        )

        metadata.to_csv(

            output_folder / "metadata.csv",

            index=False,

        )

        with open(

            output_folder / "feature_names.json",

            "w",

        ) as f:

            json.dump(

                FEATURE_ORDER,

                f,

                indent=4,

            )

        print()

        print("="*60)

        print("Dataset Saved")

        print("="*60)

        print("X :", X.shape)

        print("y :", y.shape)

        print("metadata :", len(metadata))
