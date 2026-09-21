"""FineWeb / FineWeb-2 downloader, one document per line, shuffled by seed.
Japanese is MeCab-segmented so word budgets mean the same."""
import argparse
import os
import sys

from datatrove.pipeline.readers import ParquetReader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from corpus import write_shuffled_documents  # noqa: E402


def _mecab_segmenter():
    import fugashi
    tagger = fugashi.Tagger()

    def segment(text):
        return " ".join(w.surface for w in tagger(text) if w.surface.strip())
    return segment


def documents(language, num_docs):
    if language == "eng_Latn":
        reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT", limit=num_docs)
    else:
        reader = ParquetReader(f"hf://datasets/HuggingFaceFW/fineweb-2/data/{language}/train",
                               limit=num_docs)
    segment = _mecab_segmenter() if language.startswith("jpn_") else None
    for doc in reader():
        text = doc.text.replace("\n", " ")
        if segment is not None:
            text = segment(text)
            if not text.split():
                continue
        yield text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="deu_Latn", help="FineWeb-2 config (eng_Latn = FineWeb)")
    parser.add_argument("--words", default=100_000_000, type=int, help="target whitespace word count")
    parser.add_argument("--output", default="output.txt", help="Output text file")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for shuffling")
    parser.add_argument("--words_per_line", default=200, type=int,
                        help="assumed words per document; sets the reader's document limit "
                             "(German has ~500 words/doc, 200 is safe)")
    args, _ = parser.parse_known_args()

    num_docs = args.words // args.words_per_line
    n_docs, n_words = write_shuffled_documents(
        documents(args.language, num_docs), args.output, args.random_seed,
        words=args.words, rate_limit=True)
    if n_words < args.words:
        print(f"Raise --words_per_line so the document limit (currently {num_docs}) "
              f"allows more documents.")
