"""Offset-aligned pretokenizer: the natural BPE stream grouped by spaCy word,
so P(word)=1 reproduces the plain tokenization exactly."""

import argparse
import itertools
import logging
import os
import sys

import numpy as np
import spacy
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from tags import POS_TAG_MARKER, PTB_PUNCT_TAGS, UPOS_PUNCT_TAGS  # noqa: E402
from textio import word_prefix_lines  # noqa: E402


# Always emitted bare; same inventory as the tokenizer extension.
PUNCT_UPOS = set(UPOS_PUNCT_TAGS)
PUNCT_PTB = set(PTB_PUNCT_TAGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw text split file (one document per line)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_word_ids", required=True)
    parser.add_argument("--output_word_offsets", required=True)
    parser.add_argument("--output_pos_ids", required=True)
    parser.add_argument("--output_pos_offsets", required=True)
    parser.add_argument("--tag_source", choices=["upos", "ptb"], default="upos")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--batch_size", type=int, default=2_000,
                        help="Lines per spaCy/tokenizer batch")
    parser.add_argument("--log_every", type=int, default=5_000_000,
                        help="Log progress every N source words")
    parser.add_argument("--max_words", type=int, default=0,
                        help="Only the first N whitespace words of --input (0 = all). "
                             "Same prefix rule as tokenizer/tokenize_file.py --words.")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    use_ptb = args.tag_source == "ptb"
    punct = PUNCT_PTB if use_ptb else PUNCT_UPOS

    logging.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Tokenizer {args.tokenizer!r} is not a fast tokenizer; offset "
            f"mapping (required for alignment) is unavailable.")
    backend_tok = tokenizer.backend_tokenizer

    # Prefer marker-prefixed tags; the word side must stay plain.
    unk_probe = tokenizer.unk_token_id
    _probe = POS_TAG_MARKER + ("NN" if args.tag_source == "ptb" else "NOUN")
    _pid = tokenizer.convert_tokens_to_ids(_probe)
    marker = POS_TAG_MARKER if (_pid is not None and _pid != unk_probe) else ""
    base_vocab = tokenizer.vocab_size
    logging.info(f"Tag scheme: {'marker-prefixed (inert on text)' if marker else 'plain strings'}; "
                 f"base vocab {base_vocab}, total {len(tokenizer)}")

    logging.info(f"Loading spaCy model {args.spacy_model} (tagger only)")
    # Tagger only.
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "lemmatizer"])

    # Normally one id per tag.
    tag_cache = {}
    unk_id = tokenizer.unk_token_id

    def tag_to_ids(s):
        cached = tag_cache.get(s)
        if cached is not None:
            return cached
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is None or tid == unk_id:
            ids = backend_tok.encode(s, add_special_tokens=False).ids
        else:
            ids = [tid]
        tag_cache[s] = ids
        return ids

    out_dirs = {os.path.dirname(p) or "."
                for p in (args.output_word_ids, args.output_word_offsets,
                          args.output_pos_ids, args.output_pos_offsets)}
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    tmp = {p: p + ".tmp" for p in (args.output_word_ids, args.output_word_offsets,
                                   args.output_pos_ids, args.output_pos_offsets)}
    word_ids_f = open(tmp[args.output_word_ids], "wb")
    word_off_f = open(tmp[args.output_word_offsets], "wb")
    pos_ids_f = open(tmp[args.output_pos_ids], "wb")
    pos_off_f = open(tmp[args.output_pos_offsets], "wb")

    word_cum = 0
    pos_cum = 0
    n_words = 0

    # offsets[0] = 0 sentinel
    word_off_f.write(np.array([0], dtype=np.int64).tobytes())
    pos_off_f.write(np.array([0], dtype=np.int64).tobytes())

    def flush(lines):
        nonlocal word_cum, pos_cum, n_words
        if not lines:
            return
        encs = backend_tok.encode_batch(lines, add_special_tokens=False)
        docs = nlp.pipe(lines, batch_size=len(lines))

        w_flat = []
        w_lens = []
        p_flat = []
        p_lens = []

        for text, enc, doc in zip(lines, encs, docs):
            words = [(t.idx, t.idx + len(t.text), (t.tag_ if use_ptb else t.pos_))
                     for t in doc if not t.is_space]
            n_w = len(words)
            if n_w == 0:
                continue

            # Monotonic: whitespace tokens go to the next word; spans concatenate to enc.ids.
            groups = [[] for _ in range(n_w)]
            wp = 0
            for tid, (s, _e) in zip(enc.ids, enc.offsets):
                while wp < n_w - 1 and words[wp][1] <= s:
                    wp += 1
                groups[wp].append(tid)

            for wi in range(n_w):
                ws, _we, tag = words[wi]
                toks = groups[wi]
                w_flat.extend(toks)
                w_lens.append(len(toks))

                lead = ws > 0 and text[ws - 1].isspace()
                if tag in punct:
                    tag_str = marker + tag
                else:
                    tag_str = (marker + " " + tag) if lead else (marker + tag)
                pos_ids = tag_to_ids(tag_str)
                p_flat.extend(pos_ids)
                p_lens.append(len(pos_ids))

            n_words += n_w

        if not w_lens:
            return

        if marker and w_flat and max(w_flat) >= base_vocab:
            raise RuntimeError(
                "An added (tag) token id appeared on the WORD side: the extended "
                "tokenizer is not inert on natural text, so the word side would "
                "not match the plain tokenization.  Refusing to write.")

        w_lens = np.asarray(w_lens, dtype=np.int64)
        p_lens = np.asarray(p_lens, dtype=np.int64)
        w_offsets = np.cumsum(w_lens) + word_cum
        p_offsets = np.cumsum(p_lens) + pos_cum
        word_cum = int(w_offsets[-1])
        pos_cum = int(p_offsets[-1])

        word_ids_f.write(np.asarray(w_flat, dtype=np.int32).tobytes())
        pos_ids_f.write(np.asarray(p_flat, dtype=np.int32).tobytes())
        word_off_f.write(w_offsets.tobytes())
        pos_off_f.write(p_offsets.tobytes())

    try:
        batch = []
        next_log = args.log_every
        if args.max_words:
            logging.info(f"Capping input at the first {args.max_words:,} words")
        for line in word_prefix_lines(args.input, args.max_words or None):
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
                if n_words >= next_log:
                    logging.info(f"  {n_words:,} source words processed  "
                                 f"(word subwords={word_cum:,}  "
                                 f"pos subwords={pos_cum:,})")
                    next_log += args.log_every
        flush(batch)

        for f in (word_ids_f, word_off_f, pos_ids_f, pos_off_f):
            f.close()
        for real, t in tmp.items():
            os.replace(t, real)

        logging.info(
            f"Done: {n_words:,} source words.  "
            f"word_ids={word_cum:,} ({word_cum / max(n_words,1):.2f}/src)  "
            f"pos_ids={pos_cum:,} ({pos_cum / max(n_words,1):.2f}/src)")

    except Exception:
        for f in (word_ids_f, word_off_f, pos_ids_f, pos_off_f):
            try:
                f.close()
            except Exception:
                pass
        for t in tmp.values():
            if os.path.exists(t):
                os.remove(t)
        raise


if __name__ == "__main__":
    main()
