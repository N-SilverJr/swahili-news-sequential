"""PyTorch datasets. Batches are dicts so that every model is called as model(**batch)."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Variable-length word-id sequences + labels (CNN, BiRNN, fastText-style bag)."""

    def __init__(self, seqs, y=None, word_dropout=0.0):
        self.seqs, self.y, self.word_dropout = seqs, y, word_dropout

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        if self.word_dropout > 0:            # augmentation: randomly replace words by <unk>
            s = np.where(np.random.rand(len(s)) < self.word_dropout, 1, s)
        return s, (-1 if self.y is None else int(self.y[i]))


def pad_collate(batch, min_len=5):
    """Pad to the longest sequence in the batch (min 5 so the widest CNN kernel always fits)."""
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    L = max(int(lengths.max()), min_len)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.as_tensor(s)
    return {"input_ids": ids, "lengths": lengths}, torch.tensor(ys)


class TransformerDataset(Dataset):
    """Pre-tokenised sub-word inputs for Hugging Face models."""

    def __init__(self, encodings, y=None):
        self.enc, self.y = encodings, y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, (-1 if self.y is None else int(self.y[i]))


def transformer_collate(batch, pad_id=1):
    items, ys = zip(*batch)
    L = max(len(it["input_ids"]) for it in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(items), L, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, :n] = torch.as_tensor(it["input_ids"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask}, torch.tensor(ys)
