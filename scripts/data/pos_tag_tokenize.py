"""Per-datapoint POS-tag-only tokenizer: every word replaced by its tag,
seq_length tokens per datapoint into an int64 memmap."""

import argparse
import bisect
import logging
import os
import random
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from textio import MemmapWriter  # noqa: E402

TOKEN_DTYPE = np.int64


def tsv_word_stream(input_path, tag_source="upos"):
    tag_col = 3 if tag_source == "upos" else 4
    with open(input_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                yield parts[0], parts[tag_col]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Preprocessed TSV file")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq_length", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag_source", choices=["upos", "ptb"], default="upos",
                        help="Which tag column to use: upos (17 tags) or ptb (~45 tags)")
    parser.add_argument("--random_seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    random.seed(args.random_seed)

    logging.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    seq_length = args.seq_length
    n_datapoints = 0

    with MemmapWriter(args.output, dtype=TOKEN_DTYPE, capacity=10_000_000) as out:
        buf = []
        stream = tsv_word_stream(args.input, tag_source=args.tag_source)
        eof = False

        while True:
            while len(buf) < seq_length and not eof:
                try:
                    buf.append(next(stream))
                except StopIteration:
                    eof = True
                    break
            if len(buf) < seq_length:
                break

            window = buf[:seq_length]

            # Replace every word with its POS tag
            replaced_words = [pos for _word, pos in window]

            text = " ".join(replaced_words)
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            input_ids = enc["input_ids"]
            offsets_map = enc["offset_mapping"]

            word_starts = []
            pos = 0
            for w in replaced_words:
                word_starts.append(pos)
                pos += len(w) + 1

            if len(input_ids) < seq_length:
                pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
                input_ids = list(input_ids) + [pad_id] * (seq_length - len(input_ids))
                consumed = seq_length
            else:
                input_ids = list(input_ids[:seq_length])
                last_token_start = offsets_map[seq_length - 1][0]
                word_idx = bisect.bisect_right(word_starts, last_token_start) - 1
                consumed = max(1, word_idx + 1)

            out.append(input_ids)
            n_datapoints += 1

            buf = buf[consumed:]

            if n_datapoints % 1000 == 0:
                logging.info(f"  {n_datapoints} datapoints, {out.n} tokens")

        logging.info(f"Done: {n_datapoints} datapoints, {out.n} tokens")


if __name__ == "__main__":
    main()
