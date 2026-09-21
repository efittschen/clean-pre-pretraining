"""Train the BPE tokenizer (ByteLevel + NFKC, GPT2TokenizerFast) on the first
--words whitespace words of the split."""
import argparse
import logging
import os
import sys

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from textio import word_prefix_lines  # noqa: E402


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="text input")
    parser.add_argument("--output", help="Tokenizer model output")
    parser.add_argument("--words", type=int, required=True,
                        help="train on only the first N whitespace words of the input")
    parser.add_argument("--vocab_size", type=int, default=16000, help="vocabulary size")
    parser.add_argument("--seq_length", type=int, default=128, help="model max sequence length")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    logging.info(f"Training tokenizer on first {args.words} words of {args.input}")
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    tokenizer.train_from_iterator(word_prefix_lines(args.input, args.words), trainer)

    logging.info("Saving tokenizer model")
    tokenizer.save(args.output + ".json", pretty=True)

    gpt2_tokenizer = GPT2TokenizerFast(tokenizer_file=args.output + ".json")
    gpt2_tokenizer.bos_token = "<s>"
    gpt2_tokenizer.eos_token = "</s>"
    gpt2_tokenizer.pad_token = "<pad>"
    gpt2_tokenizer.model_max_length = args.seq_length
    gpt2_tokenizer.save_pretrained(args.output)

    logging.info("Testing tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.output)

    text = "The quick brown fox jumps over the lazy dog."

    encoded = tokenizer.encode(text)
    logging.info(f"Encoded String: {encoded}")

    decoded = tokenizer.decode(encoded)
    logging.info(f"Decoded String: {decoded}")

    os.remove(args.output + ".json")
