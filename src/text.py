"""Text preprocessing and sequence representation for Swahili news.

EDA findings that motivated each step:
* Sentences are often glued together at full stops ("...yoyote.Hayo yalisemwa..."), which would
  create thousands of fake tokens such as "yoyote.hayo". We re-insert the missing space.
* Mixed case (SERIKALI vs serikali) multiplies the vocabulary, so text is lower-cased.
* Numbers (dates, scores, amounts) are mostly unique; they are mapped to a <num> token, which
  keeps the information that a number occurred (scores -> sports, amounts -> business).
"""
import re
from collections import Counter

import numpy as np

from . import config as C

GLUED = re.compile(r"(?<=[a-z\u00C0-\u024F\)\]”\"'])([.!?;:])(?=[A-Za-z\u00C0-\u024F“\"])")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
TOKEN = re.compile(r"<num>|[a-z\u00C0-\u024F]+(?:['’][a-z]+)?")

# A short list of very frequent Swahili function words (used only in one TF-IDF experiment).
SW_STOPWORDS = set("""
na ya wa kwa ni la za katika kuwa hiyo huo hilo hizo hicho hao cha vya pia hata lakini kama ili
au bila hii huu hiki hawa kwamba kuhusu baada kabla hadi tu sana zaidi ndani nje juu chini yake
yao wake wao yetu wetu letu langu yangu wangu zao zake kila mmoja moja mbili sasa leo jana
alisema amesema wakati huku ambayo ambao ambaye ambapo alikuwa amekuwa wamekuwa kutoka kati
""".split())

PAD, UNK = "<pad>", "<unk>"


def clean(text: str) -> str:
    text = GLUED.sub(r"\1 ", str(text))
    text = text.replace("\u00a0", " ").lower()
    text = NUM.sub(" <num> ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, cleaned=False):
    return TOKEN.findall(text if cleaned else clean(text))


def head_tail(tokens, max_len, head_frac=1.0):
    """Truncate a long sequence: keep the first head_frac*max_len tokens and the last rest.
    head_frac=1.0 is ordinary head truncation (news puts the key facts first)."""
    if len(tokens) <= max_len:
        return tokens
    h = int(round(max_len * head_frac))
    return tokens[:h] + (tokens[-(max_len - h):] if max_len - h > 0 else [])


class Vocab:
    """Word vocabulary built on the TRAINING split only (so validation/test OOV rates are honest)."""

    def __init__(self, token_lists, min_freq=C.MIN_FREQ, max_size=C.MAX_VOCAB):
        self.counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in self.counts.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]

    def oov_rate(self, token_lists):
        n = sum(len(t) for t in token_lists)
        return sum(1 for toks in token_lists for t in toks if t not in self.stoi) / max(n, 1)


def encode_corpus(texts, vocab, max_len=C.MAX_WORDS, head_frac=1.0):
    """List of variable-length id arrays (padding is done per batch in the collate function)."""
    out = []
    for t in texts:
        ids = vocab.encode(head_tail(tokenize(t), max_len, head_frac))
        out.append(np.array(ids if ids else [1], dtype=np.int64))
    return out


def load_fasttext_vectors(vocab, path, dim=300):
    """Initialise an embedding matrix from fastText .vec file (Grave et al., 2018).
    Words not in the file get small random vectors. Returns (matrix, coverage)."""
    import gzip
    rng = np.random.default_rng(C.SEED)
    E = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    E[0] = 0
    found = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            w, *vals = line.rstrip().split(" ")
            i = vocab.stoi.get(w)
            if i is not None and len(vals) == dim:
                E[i] = np.asarray(vals, dtype=np.float32)
                found += 1
    return E, found / (len(vocab) - 2)
