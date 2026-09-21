import argparse
import random
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input dataset file (one document per line)")
    parser.add_argument("--output_files", type=str, nargs="+", help="output files for train, dev, test splits")
    parser.add_argument("--split_sizes", type=int, nargs="+", help="sizes (in words) for train, dev, test splits")
    parser.add_argument("--seed", type=int, default=42, help="random seed for shuffling")
    args, rest = parser.parse_known_args()

    random.seed(args.seed)
    split_information = list(zip(args.output_files, args.split_sizes))

    # First pass: collect (offset, word_count) per line
    line_info = []  # (byte_offset, word_count)
    total_words = 0
    with open(args.input, "r") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            wc = len(line.split())
            if wc > 0:
                line_info.append((offset, wc))
                total_words += wc

    print(f"Total lines: {len(line_info)}, total words: {total_words}")
    assert sum(size for _, size in split_information) <= total_words, \
        f"Sum of split sizes ({sum(size for _, size in split_information)}) exceeds total words ({total_words})"

    # Shuffle lines
    random.shuffle(line_info)

    # Assign lines to splits
    with open(args.input, "r") as f_in:
        idx = 0
        for out_file, target_size in split_information:
            split_words = 0
            split_offsets = []
            while idx < len(line_info) and split_words < target_size:
                split_offsets.append(line_info[idx][0])
                split_words += line_info[idx][1]
                idx += 1

            print(f"Allocating {split_words} words ({len(split_offsets)} lines) to {out_file}")
            words_written = 0
            with open(out_file, "w") as f_out:
                for offset in split_offsets:
                    f_in.seek(offset)
                    line = f_in.readline()
                    if words_written + len(line.split()) > target_size:
                        # Truncate last line to hit exact word count
                        remaining = target_size - words_written
                        line = ' '.join(line.split()[:remaining]) + "\n"
                    f_out.write(line)
                    words_written += len(line.split())
