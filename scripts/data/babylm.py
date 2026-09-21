"""Download a BabyLM-community HF repo (raw text files per domain) into one
shuffled file, one document per line."""
import argparse
import fnmatch
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from corpus import write_shuffled_documents  # noqa: E402

_SKIP_NAMES = {"README.md", ".gitattributes", ".DS_Store", "LICENSE", "license"}


def documents(repo, data_files):
    for rel_path in data_files:
        local = hf_hub_download(repo, rel_path, repo_type="dataset")
        with open(local, "rt", errors="replace") as src:
            for line in src:
                line = line.replace("\n", " ").strip()
                if line:
                    yield line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", default=None, type=int,
                        help="Optional cap on total word count; unset = keep everything.")
    parser.add_argument("--output", default="output.txt",
                        help="Output text file (one doc per line, shuffled).")
    parser.add_argument("--random_seed", default=42, type=int,
                        help="Random seed for document-order shuffling.")
    parser.add_argument("--repo", default="BabyLM-community/BabyLM-2026-Strict-Small",
                        help="HuggingFace dataset repo id.")
    parser.add_argument("--file_pattern", default="*",
                        help="fnmatch pattern for which repo files to pull "
                             "(default '*' = all non-metadata files).")
    # Accepted but ignored: the SCons builder passes the same flag set to every
    # dataset-fetch script, and BabyLM repos have no HF configs or named splits.
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--text_field", default="text")
    args, _ = parser.parse_known_args()

    api = HfApi()
    all_files = api.list_repo_files(args.repo, repo_type="dataset")
    data_files = [
        f for f in all_files
        if os.path.basename(f) not in _SKIP_NAMES
        and fnmatch.fnmatch(os.path.basename(f), args.file_pattern)
    ]
    if not data_files:
        raise RuntimeError(
            f"No data files matched in {args.repo!r} "
            f"(pattern={args.file_pattern!r}).  All files: {all_files}")
    print(f"Pulling {len(data_files)} file(s) from {args.repo}:")
    for f in data_files:
        print(f"  {f}")

    write_shuffled_documents(documents(args.repo, data_files), args.output,
                             args.random_seed, words=args.words, rate_limit=False)
