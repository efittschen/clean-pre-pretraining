"""Text and token-stream helpers: word_prefix_lines is the one definition of
the word-budget prefix; MemmapWriter publishes .tok/.bin streams atomically."""
import logging
import os

import numpy as np


def word_prefix_lines(path, n_words=None):
    count = 0
    with open(path, "rt") as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if n_words and count + len(words) >= n_words:
                yield " ".join(words[:n_words - count])
                count = n_words
                break
            count += len(words)
            yield line.rstrip("\n")
    if n_words and count < n_words:
        logging.warning(f"Input exhausted at {count} words (< requested {n_words})")


class MemmapWriter:

    def __init__(self, output, dtype=np.int64, capacity=100_000_000):
        self.output = output
        self.dtype = np.dtype(dtype)
        out_dir = os.path.dirname(output) or "."
        os.makedirs(out_dir, exist_ok=True)
        self.tmp = os.path.join(out_dir, f".{os.path.basename(output)}.tmp_{os.getpid()}")
        self.capacity = int(capacity)
        self.n = 0
        self.mmap = np.memmap(self.tmp, dtype=self.dtype, mode="w+", shape=(self.capacity,))

    def append(self, ids):
        ids = np.asarray(ids, dtype=self.dtype)
        needed = self.n + len(ids)
        if needed > self.capacity:
            while self.capacity < needed:
                self.capacity *= 2
            del self.mmap                                   # flush and unmap
            with open(self.tmp, "ab") as f:
                f.truncate(self.capacity * self.dtype.itemsize)
            self.mmap = np.memmap(self.tmp, dtype=self.dtype, mode="r+", shape=(self.capacity,))
        self.mmap[self.n:needed] = ids
        self.n = needed

    def close(self):
        del self.mmap
        self.mmap = None
        with open(self.tmp, "ab") as f:
            f.truncate(self.n * self.dtype.itemsize)
        os.replace(self.tmp, self.output)

    def abort(self):
        self.mmap = None
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
