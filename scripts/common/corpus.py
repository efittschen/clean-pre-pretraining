"""Shared body of the corpus downloaders: stream documents, count words to
the budget, shuffle document order by seed, write."""
import os
import random
import tempfile
import time

from tqdm import tqdm


class _RateLimiter:

    def __init__(self, interval=50, threshold=200.0, grace=120):
        self.interval = interval
        self.threshold = threshold
        self.grace = grace
        self.last_time = None
        self.last_count = 0
        self.resume_after = time.time() + grace

    def check(self, count):
        if self.last_time is not None and (count - self.last_count) >= self.interval:
            now = time.time()
            elapsed = now - self.last_time
            rate = (count - self.last_count) / elapsed if elapsed > 0 else float("inf")
            if rate < self.threshold:
                sleep_time = random.uniform(120, 360)
                print(f"\nRate dropped to {rate:.1f} it/s, sleeping for {sleep_time:.0f}s...")
                time.sleep(sleep_time)
                self.resume_after = time.time() + self.grace
                self.last_time = None
            else:
                self.last_count = count
                self.last_time = time.time()
        elif self.last_time is None and time.time() >= self.resume_after:
            self.last_time = time.time()
            self.last_count = count


def write_shuffled_documents(documents, output, random_seed, words=None,
                             rate_limit=False, expected_docs=None):
    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(tmp_fd)

    line_offsets = []
    total_words = 0
    limiter = _RateLimiter() if rate_limit else None
    try:
        with open(tmp_path, "wt") as tmp_f:
            for count, text in enumerate(tqdm(documents, total=expected_docs)):
                offset = tmp_f.tell()
                tmp_f.write(text + "\n")
                line_offsets.append(offset)
                total_words += len(text.split())
                if words is not None and total_words >= words:
                    break
                if limiter is not None:
                    limiter.check(count + 1)

        print(f"Collected {len(line_offsets)} documents, {total_words} words -> {tmp_path}")
        if words is not None and total_words < words:
            print(f"WARNING: source exhausted at {total_words} words, short of the "
                  f"requested {words}.")

        # Shuffle line offsets instead of holding all text in memory.
        random.seed(random_seed)
        random.shuffle(line_offsets)

        with open(tmp_path, "rt") as tmp_f, open(output, "wt") as out_f:
            for offset in tqdm(line_offsets, desc="Writing shuffled"):
                tmp_f.seek(offset)
                out_f.write(tmp_f.readline())
    finally:
        os.unlink(tmp_path)
    return len(line_offsets), total_words
