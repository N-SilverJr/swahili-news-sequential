"""Model architectures.

Baseline (neural, but ORDER-FREE):
    FastTextBag      - averaged word (+ hashed bigram) embeddings, Joulin et al. (2017)
Neural SEQUENTIAL approaches compared in the study:
    TextCNN          - 1-D convolutions over the word sequence: LOCAL n-gram patterns (Kim, 2014)
    BiRNNAttention   - BiLSTM/BiGRU: ORDERED dependencies across the whole article, in both
                       directions, with masked additive attention pooling
    HFTextClassifier - pretrained multilingual / African transformer (XLM-R, AfroXLMR): GLOBAL
                       self-attention over sub-word tokens + knowledge from pretraining
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(input_ids):
    return (input_ids != 0)


def make_embedding(vocab_size, dim, pretrained=None, freeze=False):
    emb = nn.Embedding(vocab_size, dim, padding_idx=0)
    if pretrained is not None:
        emb.weight.data.copy_(torch.as_tensor(pretrained))
        emb.weight.requires_grad = not freeze
    return emb


# ------------------------------------------------------------------ baseline: fastText-style bag
class FastTextBag(nn.Module):
    def __init__(self, vocab_size, n_classes, dim=100, bigrams=True, buckets=200_000, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.bigrams, self.buckets = bigrams, buckets
        self.bi_emb = nn.Embedding(buckets + 1, dim, padding_idx=0) if bigrams else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, input_ids, lengths=None, **_):
        m = _mask(input_ids).float()
        vecs, cnt = (self.emb(input_ids) * m.unsqueeze(-1)).sum(1), m.sum(1, keepdim=True)
        if self.bigrams and input_ids.shape[1] > 1:
            a, b = input_ids[:, :-1], input_ids[:, 1:]
            bm = (_mask(a) & _mask(b))
            h = ((a * 1_000_003 + b) % self.buckets + 1) * bm      # hashing trick; 0 = padding
            vecs = vecs + (self.bi_emb(h) * bm.unsqueeze(-1)).sum(1)
            cnt = cnt + bm.float().sum(1, keepdim=True)
        return self.fc(self.drop(vecs / cnt.clamp_min(1)))


# ------------------------------------------------------------------ 1. TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, dim=128, kernels=(3, 4, 5), filters=128, dropout=0.5,
                 pretrained=None, freeze_emb=False):
        super().__init__()
        self.emb = make_embedding(vocab_size, dim, pretrained, freeze_emb)
        self.convs = nn.ModuleList([nn.Conv1d(dim, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels), n_classes)

    def forward(self, input_ids, lengths=None, **_):
        m = _mask(input_ids)
        x = self.drop(self.emb(input_ids)).transpose(1, 2)            # (B, dim, T)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))[:, :, :input_ids.shape[1]]            # (B, filters, T)
            h = h.masked_fill(~m.unsqueeze(1), -1e4)                  # ignore padding in max-pool
            feats.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(feats, 1)))


# ------------------------------------------------------------------ 2. BiRNN + attention
class BiRNNAttention(nn.Module):
    def __init__(self, vocab_size, n_classes, dim=128, hidden=128, layers=1, rnn="lstm",
                 pooling="attention", bidirectional=True, dropout=0.4, pretrained=None, freeze_emb=False):
        super().__init__()
        self.emb = make_embedding(vocab_size, dim, pretrained, freeze_emb)
        RNN = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn]
        self.rnn = RNN(dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        d = hidden * (2 if bidirectional else 1)
        self.pooling = pooling
        self.att_proj, self.att_v = nn.Linear(d, d), nn.Linear(d, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, n_classes)
        self.last_attention = None

    def forward(self, input_ids, lengths, **_):
        x = self.drop(self.emb(input_ids))
        packed = pack_padded_sequence(x, lengths.cpu().clamp_min(1), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=input_ids.shape[1])
        m = _mask(input_ids)
        if self.pooling == "attention":
            e = self.att_v(torch.tanh(self.att_proj(h))).squeeze(-1).masked_fill(~m, -1e4)
            a = torch.softmax(e, dim=1)
            self.last_attention = a.detach()
            z = (a.unsqueeze(-1) * h).sum(1)
        elif self.pooling == "max":
            z = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
        else:                                                         # masked mean
            z = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        return self.fc(self.drop(z))


# ------------------------------------------------------------------ 3. pretrained transformer
class HFTextClassifier(nn.Module):
    """Hugging Face encoder + classification head. tiny_random=True builds a 2-layer random BERT
    (offline smoke tests only - never for real experiments)."""

    def __init__(self, checkpoint, n_classes, tiny_random=False, vocab_size=None):
        super().__init__()
        if tiny_random:
            from transformers import BertConfig, BertForSequenceClassification
            cfg = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=64, max_position_embeddings=520, num_labels=n_classes,
                             pad_token_id=1)
            self.model = BertForSequenceClassification(cfg)
        else:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=n_classes)

    def forward(self, input_ids, attention_mask, **_):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
