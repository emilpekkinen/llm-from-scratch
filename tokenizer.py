"""Byte-level BPE tokenizer (GPT-2 style), pure Python.

Text is pre-split into "words" with a regex so merges never cross word
boundaries, then the most frequent adjacent byte pair is merged repeatedly
until the vocabulary reaches the requested size.
"""
import json
import re
from collections import Counter

# Simplified GPT-2 split pattern (stdlib `re` has no \p{L}, so ASCII classes are used)
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def _merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = dict(merges or {})  # (id_a, id_b) -> new_id, in merge order
        self._build_vocab()

    def _build_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        word_freq = Counter(PAT.findall(text))
        splits = {w: list(w.encode("utf-8")) for w in word_freq}
        self.merges = {}
        for i in range(vocab_size - 256):
            pairs = Counter()
            for w, ids in splits.items():
                f = word_freq[w]
                for p in zip(ids, ids[1:]):
                    pairs[p] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + i
            self.merges[best] = new_id
            for w, ids in splits.items():
                if len(ids) > 1:
                    splits[w] = _merge(ids, best, new_id)
            if verbose and (i + 1) % 64 == 0:
                tok = (self._bytes(best[0]) + self._bytes(best[1])).decode("utf-8", errors="replace")
                print(f"  merge {i + 1}/{vocab_size - 256}: {best} -> {new_id} {tok!r} (count {pairs[best]})")
        self._build_vocab()

    def _bytes(self, idx):
        # Used during training before self.vocab is rebuilt
        if idx < 256:
            return bytes([idx])
        for (a, b), j in self.merges.items():
            if j == idx:
                return self._bytes(a) + self._bytes(b)
        raise KeyError(idx)

    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge available in this word
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for w in PAT.findall(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls({(a, b): idx for a, b, idx in data["merges"]})
