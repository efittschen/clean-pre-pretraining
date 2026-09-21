"""spaCy + Snowball preprocess of text into a per-token TSV: word, lemma,
stem, UPOS, PTB, morph.  Chunked by byte range."""

import argparse
import logging
import os

import spacy
from nltk.stem.snowball import SnowballStemmer


def split_punct(word):
    i = 0
    while i < len(word) and not word[i].isalpha():
        i += 1
    j = len(word)
    while j > i and not word[j - 1].isalpha():
        j -= 1
    return word[:i], word[i:j], word[j:]


def snowball_of(stemmer, word):
    _, core, _ = split_punct(word)
    if not core:
        return ""
    return stemmer.stem(core.lower()) or ""


def sanitize(s):
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="0-indexed chunk index (default 0)")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Total number of chunks (default 1 = no chunking)")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading spaCy model {args.spacy_model}")
    # Keep tagger + lemmatizer; drop NER and parser for speed.
    nlp = spacy.load(args.spacy_model, disable=["ner", "parser"])
    # Attribute_ruler/morphologizer (which supplies lemma rules + morph) is
    # kept by default.  We intentionally do NOT pre-tokenize: spaCy splits
    # punctuation off words natively.
    stemmer = SnowballStemmer("english")

    tmp_dir = os.path.dirname(args.output) or "."
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f".pos_preproc_tmp_{os.getpid()}.tsv")

    file_size = os.path.getsize(args.input)
    start_offset = (file_size * args.chunk_idx) // args.num_chunks
    end_offset = (file_size * (args.chunk_idx + 1)) // args.num_chunks
    logging.info(f"Chunk {args.chunk_idx}/{args.num_chunks}: "
                 f"bytes [{start_offset}, {end_offset}) of {file_size}")

    def text_reader():
        with open(args.input, "rt") as f:
            if start_offset > 0:
                f.seek(start_offset)
                f.readline()  # skip partial leading line (belongs to previous chunk)
            while True:
                if f.tell() >= end_offset:
                    break
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    yield line

    n_tokens = 0
    try:
        with open(tmp_path, "wt") as out:
            for doc in nlp.pipe(text_reader(), batch_size=1000):
                for tok in doc:
                    if tok.is_space:
                        continue
                    word = sanitize(tok.text)
                    lemma = sanitize(tok.lemma_)
                    stem = snowball_of(stemmer, tok.text)
                    upos = tok.pos_
                    ptb = tok.tag_
                    morph = str(tok.morph)
                    out.write(f"{word}\t{lemma}\t{stem}\t{upos}\t{ptb}\t{morph}\n")
                    n_tokens += 1
                    if n_tokens % 10_000_000 == 0:
                        logging.info(f"  {n_tokens} tokens processed")

        logging.info(f"Done: {n_tokens} tokens written to {args.output}")
        os.replace(tmp_path, args.output)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main()
