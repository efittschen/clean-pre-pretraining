"""Concatenate input files in order into a single output file.

Used to stitch together chunk outputs from parallelized preprocessing.
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tmp_dir = os.path.dirname(args.output) or "."
    tmp_path = os.path.join(tmp_dir, f".concat_tmp_{os.getpid()}")
    try:
        with open(tmp_path, "wb") as out:
            for inp in args.inputs:
                with open(inp, "rb") as f:
                    shutil.copyfileobj(f, out, length=16 * 1024 * 1024)
        os.replace(tmp_path, args.output)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main()
