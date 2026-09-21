"""Tokenize a text split into the flat int64 .tok stream; --words N caps to
the word-budget prefix (common/textio.py)."""
import argparse
import logging
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from textio import MemmapWriter, word_prefix_lines  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="split text input (one document per line)")
    parser.add_argument("--tokenizer", required=True, help="tokenizer directory")
    parser.add_argument("--output", required=True, help="tokenized .tok output")
    parser.add_argument("--words", type=int, default=0,
                        help="tokenize only the first N whitespace words (0 = all)")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Encoding {'first %d words of ' % args.words if args.words else ''}file: {args.input}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total_words = 0
    with MemmapWriter(args.output, dtype=np.int64) as out:
        buffer, buffer_chars = [], 0

        def flush():
            if buffer:
                out.append(tokenizer.encode("\n".join(buffer)))

        for line in word_prefix_lines(args.input, args.words or None):
            buffer.append(line)
            buffer_chars += len(line)
            total_words += len(line.split())
            if buffer_chars >= 500_000:
                flush()
                buffer.clear()
                buffer_chars = 0
        flush()
        logging.info(f"{total_words} words -> {out.n} tokens saved to {args.output}")


if __name__ == "__main__":
    main()
