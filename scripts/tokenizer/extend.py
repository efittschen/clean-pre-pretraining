"""Extend a HuggingFace tokenizer with --extra_tokens (comma-separated,
URL-encoded); HF deduplicates against the base vocab."""

import argparse
import os
import urllib.parse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", required=True,
                        help="HF hub name or local path to the base tokenizer")
    parser.add_argument("--extra_tokens", required=True,
                        help="Comma-separated list of token strings to add. "
                             "Each entry is URL-decoded before being added so "
                             "you can pass spaces, commas, quotes, etc. safely "
                             "through the SCons command line.")
    parser.add_argument("--output", required=True,
                        help="Output tokenizer directory")
    args, _ = parser.parse_known_args()

    # URL-decode each entry so commas/spaces inside individual tokens are
    # preserved (e.g. " NNP" or "PRP$" pass through unmolested).
    extras = [urllib.parse.unquote(t) for t in args.extra_tokens.split(",") if t]
    print(f"[tokenizer_extend] Loading base tokenizer: {args.base_tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    vocab_before = len(tok)

    print(f"[tokenizer_extend] Requesting {len(extras)} extra tokens: "
          f"{extras[:6]}{' ...' if len(extras) > 6 else ''}")
    added = tok.add_tokens(extras)
    vocab_after = len(tok)
    print(f"[tokenizer_extend] {added} truly new (the rest already existed)")
    print(f"[tokenizer_extend] Vocab size: {vocab_before} -> {vocab_after}")

    os.makedirs(args.output, exist_ok=True)
    tok.save_pretrained(args.output)
    print(f"[tokenizer_extend] Saved to {args.output}")


if __name__ == "__main__":
    main()
