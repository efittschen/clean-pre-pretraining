"""The one perplexity evaluator: --stream {tok,word_ids,pos_ids} picks the
layout, --mask grammar_tags scores language predictions only."""
import json
import math
import os
import sys

import fire
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "model"))
from dataloader import GBDataset  # noqa: E402
import matrix_scales  # noqa: E402

_SUFFIX = {"tok": ".tok", "word_ids": ".word_ids.bin", "pos_ids": ".pos_ids.bin"}


def _pos_tag_strings(tag_source):
    from tags import pos_tag_extras_for
    return pos_tag_extras_for(tag_source) + pos_tag_extras_for(tag_source, marker=True)


def _grammar_tag_tail_ids(tokenizer, tag_source):
    base_vocab = tokenizer.vocab_size
    unk = tokenizer.unk_token_id
    tag_ids = set()
    for tok in _pos_tag_strings(tag_source):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == unk:
            continue
        if tid >= base_vocab:
            tag_ids.add(int(tid))
    return sorted(tag_ids)


def _resolve_data_path(checkpoint, training_dir, split, data_path, stream):
    if data_path:
        return data_path, os.path.basename(data_path)
    with open(os.path.join(training_dir, "scons_config.json")) as f:
        scons_cfg = json.load(f)
    dev_data_path = scons_cfg["dev_dataset"]
    if not split:
        return dev_data_path, "dev"
    valid = ("dev", "train", "test")
    if split not in valid:
        raise ValueError(f"Unknown split {split!r}. Must be one of {valid}")
    return os.path.join(os.path.dirname(dev_data_path), f"{split}{_SUFFIX[stream]}"), split


def _load_dataset(eval_data_path, stream, seq_length, bos_id):
    if stream == "tok":
        return GBDataset(eval_data_path, seq_length, bos_token_id=bos_id)
    # word_ids / pos_ids: int32 flat stream, cast to int64 for HF, chunked into windows.
    flat = np.asarray(np.memmap(eval_data_path, dtype=np.int32, mode="r"), dtype=np.int64)
    n = (len(flat) // seq_length) * seq_length
    return torch.from_numpy(flat[:n]).view(-1, seq_length)


def eval_perplexity(checkpoint, tokenizer_path, eval_config, output, split=None,
                    data_path=None, stream="tok", mask="none", tag_source="ptb"):
    if stream not in _SUFFIX:
        raise ValueError(f"--stream must be one of {sorted(_SUFFIX)}, got {stream!r}")
    if mask not in ("none", "grammar_tags"):
        raise ValueError(f"--mask must be 'none' or 'grammar_tags', got {mask!r}")

    with open(eval_config) as f:
        cfg = json.load(f)
    split = split or cfg.get("CONFIG", {}).get("split")
    data_path = data_path or cfg.get("CONFIG", {}).get("data_path")
    tag_source = cfg.get("CONFIG", {}).get("tag_source", tag_source)

    training_dir = os.path.dirname(checkpoint)
    with open(os.path.join(training_dir, "training_config.json")) as f:
        training_cfg = json.load(f)

    seq_length = training_cfg["CONFIG"]["SEQUENCE_LENGTH"]
    dev_tokens = training_cfg["CONFIG"]["DEV_TOKENS"]
    eval_data_path, eval_split = _resolve_data_path(checkpoint, training_dir, split,
                                                    data_path, stream)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    # Fixed forward multipliers, if this checkpoint carries a matrix_scales.json.
    # A rescaled model must be scored with them.
    matrix_scales.install(model, checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    bos_id = tokenizer.bos_token_id if training_cfg["CONFIG"].get("USE_BOS", False) else None

    dataset = _load_dataset(eval_data_path, stream, seq_length, bos_id)
    dev_sequences = dev_tokens // seq_length
    if dev_sequences > 0 and len(dataset) >= dev_sequences:
        dataset = torch.utils.data.Subset(dataset, list(range(dev_sequences)))

    mask_ids_t = None
    if mask == "grammar_tags":
        ids = _grammar_tag_tail_ids(tokenizer, tag_source)
        if ids:
            mask_ids_t = torch.tensor(ids, device=device)
        print(f"[mask] {len(ids)} genuinely-new {tag_source} tag columns masked")

    total_loss = 0.0
    total_tokens = 0
    total_loss_unmasked = 0.0
    excluded_tag_targets = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            input_ids = dataset[i].unsqueeze(0).to(device)

            if mask_ids_t is None:
                outputs = model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
                continue

            logits = model(input_ids).logits
            shift_tgt = input_ids[:, 1:]
            tgt_idx = shift_tgt.unsqueeze(-1)

            lp_unmasked = torch.log_softmax(logits[:, :-1, :], dim=-1)
            nll_unmasked = -lp_unmasked.gather(-1, tgt_idx).squeeze(-1)

            masked_logits = logits[:, :-1, :].clone()
            masked_logits[..., mask_ids_t] = float("-inf")
            lp_masked = torch.log_softmax(masked_logits, dim=-1)
            nll_masked = -lp_masked.gather(-1, tgt_idx).squeeze(-1)

            # A tag-id target has an -inf masked column (NLL +inf) and is not a
            # language prediction; drop it from BOTH numbers.
            keep = ~torch.isin(shift_tgt, mask_ids_t)
            excluded_tag_targets += int((~keep).sum().item())
            nll_masked = nll_masked[keep]
            nll_unmasked = nll_unmasked[keep]

            total_loss += float(nll_masked.sum().item())
            total_loss_unmasked += float(nll_unmasked.sum().item())
            total_tokens += int(nll_masked.numel())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    results = {
        "eval_type": "perplexity",
        "stream": stream,
        "mask": mask,
        "checkpoint": checkpoint,
        "split": eval_split,
        "data_path": eval_data_path,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }
    if mask_ids_t is not None:
        avg_u = total_loss_unmasked / total_tokens if total_tokens > 0 else float("inf")
        results.update({
            "tag_source": tag_source,
            "avg_loss_unmasked": avg_u,
            "perplexity_unmasked": math.exp(avg_u) if avg_u < 100 else float("inf"),
            "excluded_tag_targets": excluded_tag_targets,
        })

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Perplexity: {perplexity:.4f} (loss: {avg_loss:.4f}) over {total_tokens} tokens")


if __name__ == "__main__":
    fire.Fire(eval_perplexity)
