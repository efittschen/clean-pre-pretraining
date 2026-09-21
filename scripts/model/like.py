"""A fresh model from an architecture config.  --random_seed gives each
RANDOM_SEED its own body init (model_config(seeded_init=True))."""
import argparse

import torch
import transformers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output model directory")
    parser.add_argument("--architecture_like", help="model architecture to mimic")
    parser.add_argument("--random_seed", type=int, default=None, help="torch seed for the init draw")
    args, _ = parser.parse_known_args()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        transformers.set_seed(args.random_seed)

    config = transformers.AutoConfig.from_pretrained(args.architecture_like)
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(args.output)
    seed_note = f" (init seed {args.random_seed})" if args.random_seed is not None else ""
    print(f"Saved model{seed_note} to {args.output}")
