"""
parser.py

Load Apple Watch Sleep Dataset.

Author: Justin
"""

from pathlib import Path

import pandas as pd
import scipy.io as sio
from datetime import datetime
from zoneinfo import ZoneInfo   # Python >=3.9


class NightData:
    """
    Container untuk satu malam recording.
    """

    def __init__(
        self,
        subject,
        night,
        motion,
        hr,
        rec_start,
        expert_label,
        dreem_label,
    ):
        self.subject = subject
        self.night = night

        self.motion = motion
        self.hr = hr

        self.rec_start = rec_start

        self.expert_label = expert_label
        self.dreem_label = dreem_label


class SleepDatasetParser:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(self.dataset_root)

    def get_subjects(self):

        subjects = sorted(
            p for p in self.dataset_root.iterdir()
            if p.is_dir() and p.name.startswith("Bidslab")
        )

        return subjects

    def get_nights(self, subject_path):

        valid_nights = []

        for folder in sorted(subject_path.iterdir()):

        # Harus folder
            if not folder.is_dir():
                continue

        # Nama folder harus angka (1,2,3,...)
            if not folder.name.isdigit():
                continue

            motion_file = folder / "motion.csv"
            hr_file = folder / "hr.csv"
            labels_file = folder / "labels.mat"

            if (
                motion_file.exists()
                and hr_file.exists()
                and labels_file.exists()
            ):

                valid_nights.append(folder)

            else:

                print(
                    f"Missing files in {folder}. Skipping this night."
                )

        return valid_nights

    def load_motion(self, folder):

        motion_path = folder / "motion.csv"

        if not motion_path.exists() or motion_path.stat().st_size == 0:
            raise ValueError(f"Empty motion file: {motion_path}")

        motion = pd.read_csv(motion_path)
        if motion.empty:
            raise ValueError(f"Empty motion data: {motion_path}")

        motion.columns = [
            "Timestamp",
            "x",
            "y",
            "z",
        ]

        return motion

    def load_hr(self, folder):

        hr_path = folder / "hr.csv"

        if not hr_path.exists() or hr_path.stat().st_size == 0:
            raise ValueError(f"Empty HR file: {hr_path}")

        hr = pd.read_csv(
            hr_path,
            header=None,
            names=[
                "Timestamp",
                "HR",
            ],
        )
        if hr.empty:
            raise ValueError(f"Empty HR data: {hr_path}")

        return hr

    def load_labels(self, folder):

        label_path = folder / "labels.mat"

        mat = sio.loadmat(label_path)

        # String dari MATLAB
        rec_start_str = str(mat["recStart"].squeeze())

        # Format sesuai dataset
        dt = datetime.strptime(
            rec_start_str,
            "%Y-%m-%d %H:%M:%S"
        )

    # Dataset menggunakan US Eastern Time
        dt = dt.replace(
            tzinfo=ZoneInfo("America/New_York")
        )

    # Konversi ke Unix timestamp
        rec_start = dt.timestamp()

        expert = mat["expert_label"].flatten()
        dreem = mat["dreem_label"].flatten()

        return rec_start, expert, dreem

    def load_night(self, folder):

        subject = folder.parent.name

        night = folder.name

        try:
            motion = self.load_motion(folder)
            hr = self.load_hr(folder)
            rec_start, expert, dreem = self.load_labels(folder)
        except Exception as exc:
            print(f"Skipping {subject} Night {night}: {exc}")
            raise

        return NightData(
            subject,
            night,
            motion,
            hr,
            rec_start,
            expert,
            dreem,
        )

    def load_all(self, subject_filter=None):

        dataset = []

        for subject in self.get_subjects():

        # Skip subject lain jika filter diberikan
            if subject_filter is not None:
                if subject.name != subject_filter:
                    continue

            print(f"\n========== {subject.name} ==========")

            for night in self.get_nights(subject):

                print(f"Loading {subject.name} Night {night.name}")

                try:
                    data = self.load_night(night)
                except Exception:
                    continue

                dataset.append(data)

        return dataset