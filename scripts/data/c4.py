"""Stream allenai/c4 'en' into one shuffled text file, one document per
line; same contract as fw_language.py."""
import argparse
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from corpus import write_shuffled_documents  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", default=100_000_000, type=int, help="Target total word count")
    parser.add_argument("--output", default="output.txt", help="Output text file")
    parser.add_argument("--random_seed", default=42, type=int,
                        help="Random seed for document-order shuffling")
    parser.add_argument("--words_per_line", default=200, type=int,
                        help="Used only for the progress-bar estimate; output is one doc per line")
    parser.add_argument("--config", default="en",
                        help="C4 config (e.g. 'en', 'en.noblocklist', 'en.noclean')")
    args, _ = parser.parse_known_args()

    ds = load_dataset("allenai/c4", args.config, split="train", streaming=True)
    documents = (ex["text"].replace("\n", " ") for ex in ds)
    write_shuffled_documents(documents, args.output, args.random_seed, words=args.words,
                             rate_limit=True,
                             expected_docs=max(1, args.words // args.words_per_line))
