"""Emit a POS-tag stream as plain text from the preprocessed TSV, one tag
per source word."""
import argparse
import logging
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Preprocessed TSV file")
    parser.add_argument("--output", required=True, help="Tag-stream text output")
    parser.add_argument("--tag_source", choices=["upos", "ptb"], default="ptb")
    parser.add_argument("--words_per_line", type=int, default=200)
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    tag_col = 3 if args.tag_source == "upos" else 4

    tmp_dir = os.path.dirname(args.output) or "."
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f".pos_tag_text_tmp_{os.getpid()}.txt")

    n_tags = 0
    try:
        with open(args.input, "rt") as f_in, open(tmp_path, "wt") as f_out:
            line_buf = []
            for line in f_in:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5 or not parts[tag_col]:
                    continue
                line_buf.append(parts[tag_col])
                if len(line_buf) >= args.words_per_line:
                    f_out.write(" ".join(line_buf) + "\n")
                    n_tags += len(line_buf)
                    line_buf = []
            if line_buf:
                f_out.write(" ".join(line_buf) + "\n")
                n_tags += len(line_buf)
        os.replace(tmp_path, args.output)
        logging.info(f"Wrote {n_tags} tags ({args.tag_source}) to {args.output}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
