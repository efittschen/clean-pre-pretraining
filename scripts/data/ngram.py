"""Sample synthetic text from an n-gram model (n = 1..3, stupid backoff,
capped vocab) built on a preprocessed TSV."""

import argparse
import bisect
import logging
import os
import random
from collections import Counter, defaultdict


def stream_words(source_tsv):
    with open(source_tsv, "rt") as f:
        for line in f:
            w = line.split("\t", 1)[0]
            if w:
                yield w


def build_vocab(source_tsv, vocab_cap):
    freqs = Counter()
    for i, w in enumerate(stream_words(source_tsv)):
        freqs[w] += 1
        if i and i % 10_000_000 == 0:
            logging.info(f"  scanned {i} tokens, vocab so far: {len(freqs)}")
    vocab = set(w for w, _ in freqs.most_common(vocab_cap))
    logging.info(f"Full vocab: {len(freqs)}, capped vocab: {len(vocab)}")
    return vocab


def build_ngram(source_tsv, vocab, n):
    def norm(w):
        return w if w in vocab else "<UNK>"

    ctx_counts = [defaultdict(Counter) for _ in range(n)]
    history = []  # last n-1 normalized tokens
    total = 0
    for w in stream_words(source_tsv):
        w = norm(w)
        ctx_counts[0][()][w] += 1
        for k in range(1, n):
            if len(history) >= k:
                context = tuple(history[-k:])
                ctx_counts[k][context][w] += 1
        history.append(w)
        if len(history) > n - 1:
            history.pop(0)
        total += 1
        if total % 10_000_000 == 0:
            logging.info(f"  counted {total} tokens, contexts so far: " +
                         ", ".join(f"{k+1}gm={len(ctx_counts[k])}" for k in range(n)))
    return ctx_counts


def precompute_distributions(ctx_counts):
    out = [{} for _ in ctx_counts]
    for k in range(len(ctx_counts)):
        for ctx, counter in ctx_counts[k].items():
            tokens = list(counter.keys())
            cum = []
            running = 0
            for v in counter.values():
                running += v
                cum.append(running)
            out[k][ctx] = (tokens, cum, running)
    return out


def sample_from_dist(dist, rng):
    tokens, cum, total = dist
    r = rng.randrange(total)
    return tokens[bisect.bisect_right(cum, r)]


def write_unigram_vectorised(dist, target_tokens, rng, out_file, words_per_line):
    import numpy as np

    tokens, cum, total = dist
    tokens_arr = np.array(tokens, dtype=object)
    cum_arr = np.array(cum, dtype=np.int64)

    np_rng = np.random.default_rng(rng.getrandbits(63))
    batch_size = 10_000_000
    remaining = target_tokens
    written = 0
    leftover = []  # tokens not yet flushed to a complete line

    while remaining > 0:
        b = min(batch_size, remaining)
        rs = np_rng.integers(0, total, size=b)
        indices = np.searchsorted(cum_arr, rs, side="right")
        batch_tokens = tokens_arr[indices].tolist()

        # Combine with leftover and flush full lines
        all_tokens = leftover + batch_tokens
        n_full_lines = len(all_tokens) // words_per_line
        for line_idx in range(n_full_lines):
            start = line_idx * words_per_line
            out_file.write(" ".join(all_tokens[start:start + words_per_line]) + "\n")
        leftover = all_tokens[n_full_lines * words_per_line:]

        written += b
        remaining -= b
        logging.info(f"  sampled {written}/{target_tokens} tokens")

    if leftover:
        out_file.write(" ".join(leftover) + "\n")


def write_ngram_stream(precomputed, n, target_tokens, rng, out_file, words_per_line):
    fallback = precomputed[0][()]
    history = []  # last n-1 tokens
    line_buffer = []
    log_interval = max(1_000_000, target_tokens // 100)

    for i in range(target_tokens):
        chosen = None
        max_k = min(n - 1, len(history))
        for k in range(max_k, 0, -1):
            context = tuple(history[-k:])
            ctx_dict = precomputed[k]
            if context in ctx_dict:
                chosen = sample_from_dist(ctx_dict[context], rng)
                break
        if chosen is None:
            chosen = sample_from_dist(fallback, rng)

        line_buffer.append(chosen)
        history.append(chosen)
        if len(history) > n - 1:
            history.pop(0)

        if len(line_buffer) >= words_per_line:
            out_file.write(" ".join(line_buffer) + "\n")
            line_buffer.clear()

        if (i + 1) % log_interval == 0:
            logging.info(f"  sampled {i + 1}/{target_tokens} tokens")

    if line_buffer:
        out_file.write(" ".join(line_buffer) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_tsv", required=True,
                        help="Preprocessed TSV (word \\t lemma \\t stem \\t UPOS \\t PTB \\t morph)")
    parser.add_argument("--n", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--target_words", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab_cap", type=int, default=100_000)
    parser.add_argument("--words_per_line", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = random.Random(args.random_seed)

    logging.info(f"First pass: counting word frequencies (vocab cap {args.vocab_cap})")
    vocab = build_vocab(args.source_tsv, args.vocab_cap)

    logging.info(f"Second pass: building {args.n}-gram counts")
    ctx_counts = build_ngram(args.source_tsv, vocab, args.n)
    for k in range(args.n):
        logging.info(f"  {k+1}-gram contexts: {len(ctx_counts[k])}")

    logging.info("Precomputing per-context cumulative distributions")
    precomputed = precompute_distributions(ctx_counts)
    del ctx_counts

    logging.info(f"Sampling {args.target_words} tokens with seed {args.random_seed}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "wt") as f_out:
        if args.n == 1:
            write_unigram_vectorised(precomputed[0][()], args.target_words, rng,
                                     f_out, args.words_per_line)
        else:
            write_ngram_stream(precomputed, args.n, args.target_words, rng,
                               f_out, args.words_per_line)
    os.replace(tmp, args.output)
    logging.info("Done")


if __name__ == "__main__":
    main()
