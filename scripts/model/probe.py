"""Checkpoint-probe machinery for train.py (CONFIG.CKPT_PROBE): per-token dev
log-likelihood by position and frequency bin."""
import json
import os

import numpy as np
import torch

# log10-frequency bin edges: bin0 = f<1e-7 (incl. unseen), bin k = [10^(k-8), 10^(k-7)), bin7 = f>=1e-1
_FREQ_EDGES = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]


def build_freq_bins(train_data, vocab_size):
    toks = np.asarray(np.memmap(train_data, dtype=np.int64, mode="r"))
    counts = np.bincount(toks, minlength=vocab_size).astype(np.float64)
    total = counts.sum()
    logf = np.full(vocab_size, -99.0)
    nz = counts > 0
    logf[nz] = np.log10(counts[nz] / total)
    bin_of = np.digitize(logf, _FREQ_EDGES)          # 0..7; -99 -> 0
    mean_logf = [float(np.mean(logf[(bin_of == b) & nz])) if np.any((bin_of == b) & nz) else None
                 for b in range(len(_FREQ_EDGES) + 1)]
    return bin_of, {"edges_log10f": _FREQ_EDGES, "bin_mean_log10f": mean_logf,
                    "train_total_tokens": int(total), "vocab_size": int(vocab_size)}


def run_probe(model, step, eval_dataset, seq_length, bin_of, bin_meta, out_dir, batch_size=4):
    was_training = model.training
    model.eval()
    n_bins = len(_FREQ_EDGES) + 1
    P = seq_length - 1                                   # target positions 1..seq-1
    pos_s = np.zeros(P); pos_s2 = np.zeros(P); pos_n = np.zeros(P)
    bin_s = np.zeros(n_bins); bin_s2 = np.zeros(n_bins); bin_n = np.zeros(n_bins)
    tot_s = 0.0; tot_s2 = 0.0; tot_n = 0
    bin_lut = torch.from_numpy(bin_of)
    dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda xs: torch.stack(xs))
    with torch.no_grad():
        for batch in dl:
            input_ids = batch.to(model.device)
            logits = model(input_ids=input_ids).logits[:, :-1].float()
            targets = input_ids[:, 1:]
            ll = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            ll_np = ll.cpu().numpy().astype(np.float64)          # [B, P]
            tgt_np = targets.cpu().numpy()
            B = ll_np.shape[0]
            pos_s += ll_np.sum(0); pos_s2 += (ll_np ** 2).sum(0); pos_n += B
            b = bin_lut[tgt_np.reshape(-1)].numpy()
            flat = ll_np.reshape(-1)
            np.add.at(bin_s, b, flat); np.add.at(bin_s2, b, flat ** 2)
            np.add.at(bin_n, b, 1)
            tot_s += flat.sum(); tot_s2 += (flat ** 2).sum(); tot_n += flat.size
    if was_training:
        model.train()
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "step": int(step),
        "overall": {"n": int(tot_n), "sum_ll": tot_s, "sum_ll2": tot_s2,
                    "mean_ll": tot_s / max(tot_n, 1), "ppl": float(np.exp(-tot_s / max(tot_n, 1)))},
        "by_position": {"target_positions": "1..seq_length-1",
                        "n": pos_n.tolist(), "sum_ll": pos_s.tolist(), "sum_ll2": pos_s2.tolist()},
        "by_freq_bin": {**bin_meta, "n": bin_n.tolist(), "sum_ll": bin_s.tolist(), "sum_ll2": bin_s2.tolist()},
    }
    path = os.path.join(out_dir, f"step_{int(step):06d}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[probe] step {step}: dev ppl {out['overall']['ppl']:.3f} -> {path}", flush=True)


