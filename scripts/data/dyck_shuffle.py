import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from grammar import generate_shuff_dyck_txt_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("num_symbols", type=int, default=64, help="Number of symbols")
    parser.add_argument("n", type=int, default=100000, help="Number of sequences")
    parser.add_argument("target_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("p", default=0.5, type=float, help="Probability of new bracket") # shuffle only
    args, rest = parser.parse_known_args()

    generate_shuff_dyck_txt_file(
            args.output,
            num_symbols=args.num_symbols,
            n=args.n,
            target_length=args.target_length,
            p=args.p,
        )

