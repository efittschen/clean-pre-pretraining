
import argparse
import json
import pathlib
import pickle
import warnings
from tqdm import tqdm
import torch
import transformers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output model directory")
    parser.add_argument("--input_model", help="model")
    parser.add_argument("--tokenizer", help="tokenizer path (if provided, resizes embeddings to match)", default=None)
    parser.add_argument("--flag", help="flag file to touch on completion", default=None)
    parser.add_argument("--std", help="standard deviation for new embeddings", type=float, default=0.02)
    args, rest = parser.parse_known_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.input_model)

    if args.tokenizer is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        # Update model config special token IDs to match the new tokenizer
        fallback_id = tokenizer.eos_token_id
        for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'decoder_start_token_id', 'sep_token_id', 'cls_token_id']:
            old_id = getattr(model.config, attr, None)
            new_id = getattr(tokenizer, attr, None)
            if old_id is not None and new_id is None:
                warnings.warn(f"Model has {attr}={old_id} but new tokenizer does not define it. Falling back to eos_token_id={fallback_id}.")
                new_id = fallback_id
            setattr(model.config, attr, new_id)

    # Reinitialise input embeddings
    old_input_emb = model.get_input_embeddings()
    print(f"Input embeddings before reset: mean norm={old_input_emb.weight.data.norm(dim=1).mean():.4f}, std={old_input_emb.weight.data.std():.4f}, shape={list(old_input_emb.weight.data.shape)}")
    new_input_emb = torch.nn.Embedding(old_input_emb.num_embeddings, old_input_emb.embedding_dim)
    new_input_emb.weight.data.normal_(mean=0.0, std=args.std)
    model.set_input_embeddings(new_input_emb)
    print(f"Input embeddings after reset:  mean norm={new_input_emb.weight.data.norm(dim=1).mean():.4f}, std={new_input_emb.weight.data.std():.4f}")

    # Reinitialise output head
    old_output_emb = model.get_output_embeddings()
    if old_output_emb is not None:
        print(f"Output head before reset: mean norm={old_output_emb.weight.data.norm(dim=1).mean():.4f}, std={old_output_emb.weight.data.std():.4f}, shape={list(old_output_emb.weight.data.shape)}")
        new_output_emb = torch.nn.Linear(old_output_emb.in_features, old_output_emb.out_features, bias=old_output_emb.bias is not None)
        new_output_emb.weight.data.normal_(mean=0.0, std=args.std)
        model.set_output_embeddings(new_output_emb)
        print(f"Output head after reset:  mean norm={new_output_emb.weight.data.norm(dim=1).mean():.4f}, std={new_output_emb.weight.data.std():.4f}")

    model.save_pretrained(args.output)
    if args.tokenizer is not None:
        tokenizer.save_pretrained(args.output)
    print(f"Saved model to {args.output}")

    if args.flag is not None:
        pathlib.Path(args.flag).touch()