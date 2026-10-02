"""Loading Train/Test, normalising labels and building the shared stratified split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_labelled(path=None) -> pd.DataFrame:
    """Train.csv -> [id, content, label]. Labels are lower-cased: the raw file mixes
    'Kitaifa' and 'michezo', while SampleSubmission.csv uses lower case."""
    df = pd.read_csv(path or C.DATA_DIR / "Train.csv")
    df = df.rename(columns={C.LABEL_COL: "label_raw"})
    df["label"] = df["label_raw"].astype(str).str.strip().str.lower()
    df[C.TEXT_COL] = df[C.TEXT_COL].fillna("").astype(str)
    unknown = set(df.label) - set(C.CLASSES)
    if unknown:
        raise ValueError(f"Unexpected labels {unknown}; update config.CLASSES")
    return df[[C.ID_COL, C.TEXT_COL, "label", "label_raw"]]


def load_unlabelled(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.DATA_DIR / "Test.csv")
    df[C.TEXT_COL] = df[C.TEXT_COL].fillna("").astype(str)
    return df.head(100) if C.QUICK else df


def get_splits(rebuild=False) -> pd.DataFrame:
    """Stratified 70 / 15 / 15 train / val / test split saved to data/splits.csv (ids only).

    Member 1 creates it once and commits it; everyone else loads the same file. Zindi's Test.csv
    is unlabelled, so our labelled test split is carved out of Train.csv and touched once per model.
    """
    df = load_labelled()
    if C.SPLITS_PATH.exists() and not rebuild:
        s = pd.read_csv(C.SPLITS_PATH)
        df = df.merge(s[[C.ID_COL, "split"]], on=C.ID_COL, how="inner")
    else:
        trval, test = train_test_split(df, test_size=C.TEST_SIZE, stratify=df.label, random_state=C.SEED)
        tr, val = train_test_split(trval, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
                                   stratify=trval.label, random_state=C.SEED)
        df = pd.concat([tr.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
        df[[C.ID_COL, "label", "split"]].to_csv(C.SPLITS_PATH, index=False)
        print(f"Saved new split to {C.SPLITS_PATH} (commit this file!)")
    print("Split sizes:", df.split.value_counts().to_dict())
    return df.reset_index(drop=True)


def split_frames(df):
    out = tuple(df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if C.QUICK:   # smoke-test mode: small subsets so every notebook runs in minutes on a CPU
        out = tuple(pd.concat([d[d.label == c].head(max(2, n * int((d.label == c).sum()) // len(d)))
                               for c in C.CLASSES]).reset_index(drop=True) for d, n in zip(out, (400, 150, 150)))
    return out


def encode_labels(labels):
    to_id = {c: i for i, c in enumerate(C.CLASSES)}
    return np.array([to_id[l] for l in labels], dtype=np.int64)
