"""Graft a donor's weights into a fresh host per the config's graft= spec;
embeddings and lm_head are never copied."""
import argparse
import json
import os

import torch
import transformers


def copy_head(sd_donor, sd_new, layer, head, head_dim, with_ln=False):
    p = "model.layers.%d." % layer
    lo, hi = head * head_dim, (head + 1) * head_dim
    n = 0
    for proj in ("q_proj", "k_proj", "v_proj"):
        k = p + "self_attn." + proj + ".weight"
        if k in sd_donor and k in sd_new:
            sd_new[k][lo:hi, :] = sd_donor[k][lo:hi, :]
            n += 1
    k = p + "self_attn.o_proj.weight"
    if k in sd_donor and k in sd_new:
        sd_new[k][:, lo:hi] = sd_donor[k][:, lo:hi]
        n += 1
    if with_ln:
        k = p + "input_layernorm.weight"
        if k in sd_donor and k in sd_new:
            sd_new[k][:] = sd_donor[k]
            n += 1
    return n


def copy_layer(sd_donor, sd_new, layer):
    p = "model.layers.%d." % layer
    n = 0
    for k in sd_new:
        if k.startswith(p) and k in sd_donor:
            sd_new[k][...] = sd_donor[k]
            n += 1
    return n


def copy_mlp(sd_donor, sd_new, layer):
    p = "model.layers.%d." % layer
    n = 0
    for k in sd_new:
        if k.startswith(p) and ("mlp." in k or "post_attention_layernorm" in k) and k in sd_donor:
            sd_new[k][...] = sd_donor[k]
            n += 1
    return n


def copy_body(sd_donor, sd_new):
    n = 0
    for k in sd_new:
        if k.startswith("model.embed_tokens") or k.startswith("lm_head"):
            continue
        if k in sd_donor and sd_donor[k].shape == sd_new[k].shape:
            sd_new[k][...] = sd_donor[k]
            n += 1
    return n


def induction_score(model, n_layers, n_heads, vocab, N=128, n_seq=16, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    core = torch.randint(0, vocab, (n_seq, N), generator=g)
    ids = torch.cat([core, core], dim=1)
    model.set_attn_implementation("eager")  # sdpa returns attentions=None
    model.eval()
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    assert out.attentions is not None and out.attentions[0] is not None
    scores = torch.zeros(n_layers, n_heads)
    q = torch.arange(N, 2 * N)
    k = q - N + 1
    for li, att in enumerate(out.attentions):
        scores[li] = att[:, :, q, k].mean(dim=(0, 2))
    return scores


def apply_spec(sd_donor, sd_new, spec, head_dim):
    n, did = 0, []
    if spec.get("body"):
        c = copy_body(sd_donor, sd_new)
        n += c; did.append("body(%d)" % c)
    for layer in spec.get("layers", []):
        c = copy_layer(sd_donor, sd_new, layer)
        n += c; did.append("layer%d(%d)" % (layer, c))
    for layer in spec.get("mlps", []):
        c = copy_mlp(sd_donor, sd_new, layer)
        n += c; did.append("mlp%d(%d)" % (layer, c))
    for layer, h in spec.get("heads", []):
        c = copy_head(sd_donor, sd_new, layer, h, head_dim,
                      spec.get("with_layernorm", False))
        n += c; did.append("L%dH%d(%d)" % (layer, h, c))
    return n, did


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="path to the graft spec JSON")
    ap.add_argument("--output", required=True)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--flag", default=None)
    a = ap.parse_args()

    spec = json.load(open(a.spec))

    cfg = transformers.AutoConfig.from_pretrained(spec["arch"])
    if spec.get("attention_bias"):
        cfg.attention_bias = True
    torch.manual_seed(a.random_seed)
    transformers.set_seed(a.random_seed)
    host = transformers.AutoModelForCausalLM.from_config(cfg)
    sd_new = host.state_dict()
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    did, n = [], 0
    if spec.get("donor") and (spec.get("heads") or spec.get("layers")
                              or spec.get("mlps") or spec.get("body")):
        sd_donor = transformers.AutoModelForCausalLM.from_pretrained(
            spec["donor"]).state_dict()
        n, did = apply_spec(sd_donor, sd_new, spec, head_dim)
    host.load_state_dict(sd_new)

    os.makedirs(a.output, exist_ok=True)
    host.save_pretrained(a.output)

    # The gate: does the built model actually show induction?  A graft that did
    # not take reads the same as the baseline here.
    gate = transformers.AutoModelForCausalLM.from_pretrained(a.output)
    gate.set_attn_implementation("eager")
    gate.eval()
    sc = induction_score(gate, cfg.num_hidden_layers, cfg.num_attention_heads,
                         cfg.vocab_size)
    report = {
        "spec": spec,
        "random_seed": a.random_seed,
        "tensors_copied": n,
        "copied": did,
        "induction_max": float(sc.max()),
        "n_heads_gt_0.2": int((sc > 0.2).sum()),
    }
    with open(os.path.join(a.output, "graft_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[graft] %s: %d tensors %s | induction_max %.4f, heads>0.2: %d"
          % (os.path.basename(a.output.rstrip("/")), n, " ".join(did) or "(nothing)",
             report["induction_max"], report["n_heads_gt_0.2"]), flush=True)

    if a.flag:
        os.makedirs(os.path.dirname(os.path.abspath(a.flag)), exist_ok=True)
        with open(a.flag, "w") as f:
            f.write("graft complete\n")


if __name__ == "__main__":
    main()
