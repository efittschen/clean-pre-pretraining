"""Construct prev-token + induction circuits directly in a fresh host.
2b is live; 3 (mirrored MLP0) was never run."""
import sys, math, torch, transformers
sys.path.insert(0, "scripts")
from graft import induction_score
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def _hidden(model, ids):
    with torch.no_grad(): return model(ids, output_hidden_states=True).hidden_states

def copy_loss(model, N=128, B=8):
    core = torch.randint(0, 16000, (B, N)); ids = torch.cat([core, core], 1)
    with torch.no_grad():
        lg = model(ids).logits[:, :-1]; tgt = ids[:, 1:]
        l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(B, -1)
        acc = (lg.argmax(-1) == tgt).float()
    return l[:, :N - 1].mean().item(), l[:, N:].mean().item(), acc[:, N:].mean().item()

def diagnose(model, LP=1, HP=0, LI=5, HI=0, N=128, B=8):
    core = torch.randint(0, 16000, (B, N)); ids = torch.cat([core, core], 1); T = 2 * N; pos = torch.arange(T)[None]; mask = torch.full((T, T), float("-inf")).triu(1)
    hs = _hidden(model, ids); res = {}
    with torch.no_grad():
        for l, h, name in [(LP, HP, "prev"), (LI, HI, "ind")]:
            a = model.model.layers[l].self_attn; ln = model.model.layers[l].input_layernorm(hs[l])
            q = a.q_proj(ln).view(B, T, 12, 64).transpose(1, 2); k = a.k_proj(ln).view(B, T, 12, 64).transpose(1, 2)
            cos, sin = model.model.rotary_emb(hs[l], pos); q, k = apply_rotary_pos_emb(q, k, cos, sin)
            lg = q[:, h] @ k[:, h].transpose(-1, -2) / 8; att = torch.softmax(lg + mask, -1); qs = torch.arange(N, T)
            if name == "prev": res["prev_score"] = att[:, 1:, :][:, torch.arange(T - 1), torch.arange(T - 1)].mean().item()
            else: res.update(ind_score=att[:, qs, qs - N + 1].mean().item(), same=lg[:, qs, qs - N + 1].mean().item(), other_std=lg[:, qs, :N].std().item())
    return res

def _pairs(HD, g, active, amp_active, amp_idle, theta, aim=True):
    half = HD // 2
    u = torch.zeros(HD)
    k = torch.zeros(HD)
    act = set(int(a) for a in active)
    for j in range(half):
        phi = float(torch.rand(1, generator=g)) * 2 * math.pi          # query phase: always random
        if j in act:
            A = float(amp_active[j]) if torch.is_tensor(amp_active) else float(amp_active)
            B = A
            psi = phi + (float(theta[j]) if aim else 0.0)              # key leads by one position
        else:
            lo = 0.1 * amp_idle
            A = float(torch.exp(torch.rand(1, generator=g) * math.log(amp_idle / lo)) * lo)
            B = float(torch.exp(torch.rand(1, generator=g) * math.log(amp_idle / lo)) * lo)
            psi = float(torch.rand(1, generator=g)) * 2 * math.pi      # unaligned -> cancels
        u[j], u[j + half] = A * math.cos(phi), A * math.sin(phi)
        k[j], k[j + half] = B * math.cos(psi), B * math.sin(psi)
    return u, k


def build2b(model, seed=0, n_align=768, rho=0.8, tilt=3.0, a_prev=8.0, self_logit=10.0,
            target_logit=12.0, slow_pairs=range(20, 32), s_o=4.0, s_oi=4.0,
            n_prev_pairs=8, idle_prev=1.0, idle_cmp=1.0, pc_align=False, reader="bottom",
            verbose=True):
    g = torch.Generator().manual_seed(seed)
    cfg = model.config; D, HD = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads
    half = HD // 2
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    Bt, Bp, Bo, Bs = Q[:, 1:65], Q[:, 65:129], Q[:, 129:193], Q[:, 193:257]
    sd = model.state_dict(); L = "model.layers.%d."
    ids = torch.randint(0, cfg.vocab_size, (4, 256), generator=g)
    theta = 10000.0 ** (-torch.arange(half).float() / half)

    # --- MLP0 ---
    G = sd[L % 0 + "mlp.gate_proj.weight"]; U = sd[L % 0 + "mlp.up_proj.weight"]
    idx = torch.arange(n_align)
    gn = torch.nn.functional.normalize(G[idx], dim=1)
    noise = torch.randn(n_align, D, generator=g)
    noise = torch.nn.functional.normalize(noise - (noise * gn).sum(1, keepdim=True) * gn, dim=1)
    un = rho * gn + math.sqrt(1 - rho ** 2) * noise
    U[idx] = un * U[idx].norm(dim=1, keepdim=True)
    if tilt > 0:
        Wd = sd[L % 0 + "mlp.down_proj.weight"]
        c0 = Q[:, 300]
        cols = Wd[:, idx]; norms = cols.norm(dim=0, keepdim=True)
        cols = cols + tilt * norms * c0[:, None]
        Wd[:, idx] = cols * (norms / cols.norm(dim=0, keepdim=True))
    model.load_state_dict(sd)

    # --- L0 self heads ---
    hs = _hidden(model, ids)
    with torch.no_grad(): ln0 = model.model.layers[0].input_layernorm(hs[0])
    Vpc = None
    if pc_align:
        with torch.no_grad():
            X = ln0.reshape(-1, D); X = X - X.mean(0)
            Vpc = torch.linalg.svd(X, full_matrices=False)[2][:HD]      # HD x D
    for h in (0, 1):
        if pc_align:            # so the two heads differ
            P = torch.linalg.qr(torch.randn(HD, HD, generator=g))[0] @ Vpc
        else:
            P = torch.linalg.qr(torch.randn(D, HD, generator=g))[0].T
        with torch.no_grad(): pn2 = (ln0 @ P.T).pow(2).sum(-1).mean().item()
        gam = math.sqrt(self_logit * math.sqrt(HD) / pn2)
        r = slice(h * HD, (h + 1) * HD)
        sd[L % 0 + "self_attn.q_proj.weight"][r] += gam * P
        sd[L % 0 + "self_attn.k_proj.weight"][r] += gam * P
        sd[L % 0 + "self_attn.v_proj.weight"][r] += Bt.T
        sd[L % 0 + "self_attn.o_proj.weight"][:, r] += Bs * (s_o / 2)
    model.load_state_dict(sd)

    # --- L1 prev heads: signal in the fastest n_prev_pairs ---
    hs = _hidden(model, ids)
    with torch.no_grad():
        ln1 = model.model.layers[1].input_layernorm(hs[1])
        nu = ln1.reshape(-1, D).mean(0); nu_hat = nu / nu.norm()
        pc = ln1 @ nu_hat; m_c, s_c = pc.mean().item(), pc.std().item()
    for h in (0, 1):
        u, ku = _pairs(HD, g, range(n_prev_pairs), a_prev, a_prev * idle_prev, theta, aim=True)
        r = slice(h * HD, (h + 1) * HD)
        sd[L % 1 + "self_attn.q_proj.weight"][r] += u[:, None] * nu_hat[None, :] / m_c
        sd[L % 1 + "self_attn.k_proj.weight"][r] += ku[:, None] * nu_hat[None, :] / m_c
        sd[L % 1 + "self_attn.v_proj.weight"][r] += Bs.T
        sd[L % 1 + "self_attn.o_proj.weight"][:, r] += Bp * (s_o / 2)
    model.load_state_dict(sd)
    hs = _hidden(model, ids)
    with torch.no_grad():
        lnI = model.model.layers[5].input_layernorm(hs[5])
        ratio = (lnI @ Bs).norm(dim=-1).mean() / ((lnI @ Bp).norm(dim=-1).mean() + 1e-9)
    for h in (0, 1):
        sd[L % 1 + "self_attn.o_proj.weight"][:, h * HD:(h + 1) * HD] *= ratio
    model.load_state_dict(sd)

    # --- L5 comparator: signal in the slow pairs ---
    hs = _hidden(model, ids)
    with torch.no_grad():
        lnI = model.model.layers[5].input_layernorm(hs[5]); qc = lnI @ Bs; kc = lnI @ Bp
        Mq = qc.reshape(-1, 64); Mk = kc.reshape(-1, 64)
        M = Mq.T @ Mq / len(Mq) + Mk.T @ Mk / len(Mk)
        n_coord = 2 * len(list(slow_pairs))
        if reader == "top":
            Qc = Mq - Mq.mean(0); Kc = Mk - Mk.mean(0)
            Uc = torch.linalg.eigh(Qc.T @ Qc / len(Qc) + Kc.T @ Kc / len(Kc))[1][:, -n_coord:]
        elif reader == "fold":
            Uc = torch.nn.functional.normalize(torch.randn(64, n_coord, generator=g), dim=0)
        elif reader == "matched":
            # top eigenvectors of the symmetrised cross-moment
            A = qc[:, :-1].reshape(-1, 64); B = kc[:, 1:].reshape(-1, 64)
            Cx = (A.T @ B + B.T @ A) / (2 * len(A))
            Uc = torch.linalg.eigh(Cx)[1][:, -n_coord:]
        else:
            Uc = torch.linalg.eigh(M)[1][:, :n_coord]
        dots = ((qc[:, :-1] @ Uc) * (kc[:, 1:] @ Uc)).sum(-1)
    scale = math.sqrt(target_logit * math.sqrt(HD) / max(dots.mean().item(), 1e-6))
    Wq = torch.zeros(HD, D); Wk = torch.zeros(HD, D)
    for i, p in enumerate(slow_pairs):                       # signal pairs
        Wq[p] = Bs @ Uc[:, 2 * i] * scale; Wq[p + half] = Bs @ Uc[:, 2 * i + 1] * scale
        Wk[p] = Bp @ Uc[:, 2 * i] * scale; Wk[p + half] = Bp @ Uc[:, 2 * i + 1] * scale
    idle = [p for p in range(half) if p not in set(slow_pairs)]
    if idle_cmp > 0:                                         # random content in the fast pairs
        amp = idle_cmp * scale * float(Uc.abs().mean())
        for p in idle:
            for W in (Wq, Wk):
                for row in (p, p + half):
                    v = torch.randn(D, generator=g)
                    W[row] = v / v.norm() * amp * float(torch.exp(torch.rand(1, generator=g) * math.log(10.0)) * 0.316)
    r = slice(0, HD)
    sd[L % 5 + "self_attn.q_proj.weight"][r] += Wq
    sd[L % 5 + "self_attn.k_proj.weight"][r] += Wk
    sd[L % 5 + "self_attn.v_proj.weight"][r] += Bs.T
    sd[L % 5 + "self_attn.o_proj.weight"][:, r] += Bo * s_oi
    model.load_state_dict(sd)
    if verbose:
        print("  v2b: carrier m_c %.2f +- %.0f%% (SNR %.1f)  prev rebalance x%.2f  cmp scale %.2f"
              % (m_c, 100 * s_c / m_c, m_c / s_c, ratio, scale), flush=True)
    return model


# ------------------------------------------------------------------------
# v3 -- NEVER RUN
# ------------------------------------------------------------------------


def empirical_cos(path, layer=0):
    sd = load_file(path)
    return F.cosine_similarity(sd["model.layers.%d.mlp.gate_proj.weight" % layer].float(),
                               sd["model.layers.%d.mlp.up_proj.weight" % layer].float(), dim=1)


def mlp0_mirrored(model, g, c0, tilt, cos_pool, verbose=True):
    sd = model.state_dict()
    G = sd["model.layers.0.mlp.gate_proj.weight"]
    U = sd["model.layers.0.mlp.up_proj.weight"]
    Wd = sd["model.layers.0.mlp.down_proj.weight"]
    n, D = G.shape

    rho = cos_pool[torch.randperm(len(cos_pool), generator=g)[:n]].clone()
    if len(cos_pool) < n:                                     # resample with replacement if short
        rho = cos_pool[torch.randint(len(cos_pool), (n,), generator=g)]

    # --- up rows at arccos(rho) from gate rows, norm kept ---
    gn = F.normalize(G, dim=1)
    noise = torch.randn(n, D, generator=g)
    noise = F.normalize(noise - (noise * gn).sum(1, keepdim=True) * gn, dim=1)
    un = rho[:, None] * gn + torch.sqrt(1 - rho[:, None] ** 2) * noise
    U.copy_(un * U.norm(dim=1, keepdim=True))

    # --- write columns tilt along c0 by the SIGNED cosine ---
    norms = Wd.norm(dim=0, keepdim=True)
    cols = Wd + tilt * rho[None, :] * norms * c0[:, None]
    Wd.copy_(cols * (norms / cols.norm(dim=0, keepdim=True)))

    model.load_state_dict(sd)
    if verbose:
        c = F.cosine_similarity(G, U, dim=1)
        proj = F.normalize(Wd, dim=0).T @ c0
        print("  MLP0 mirrored: cos sd %.4f (target 0.133)  mean|col.c0| %.4f (target 0.066)  "
              "corr %.3f (target 0.787)" % (c.std(), proj.abs().mean(),
                                            float(torch.corrcoef(torch.stack([c, proj]))[0, 1])))
    return rho


def build3(model, seed=0, tilt=1.0, cos_pool=None, a_prev=8.0, self_logit=10.0, target_logit=12.0,
           slow_pairs=range(20, 32), s_o=4.0, s_oi=4.0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    cfg = model.config; D, HD = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    Bt, Bp, Bo, Bs = Q[:, 1:65], Q[:, 65:129], Q[:, 129:193], Q[:, 193:257]
    c0 = Q[:, 300]
    sd = model.state_dict(); L = "model.layers.%d."
    ids = torch.randint(0, cfg.vocab_size, (4, 256), generator=g)

    if cos_pool is None:
        raise ValueError(
            "build3 mirrors a trained MLP0, so it needs statistics to mirror: pass "
            "cos_pool=, or set reference=<model.safetensors> in the config spec.")
    mlp0_mirrored(model, g, c0, tilt, cos_pool, verbose=verbose)
    sd = model.state_dict()

    # --- everything below is verbatim build2 ---
    hs = _hidden(model, ids)
    with torch.no_grad(): ln0 = model.model.layers[0].input_layernorm(hs[0])
    for h in (0, 1):
        P = torch.linalg.qr(torch.randn(D, HD, generator=g))[0].T
        with torch.no_grad(): pn2 = (ln0 @ P.T).pow(2).sum(-1).mean().item()
        gam = math.sqrt(self_logit * math.sqrt(HD) / pn2)
        r = slice(h * HD, (h + 1) * HD)
        sd[L % 0 + "self_attn.q_proj.weight"][r] = gam * P
        sd[L % 0 + "self_attn.k_proj.weight"][r] = gam * P
        sd[L % 0 + "self_attn.v_proj.weight"][r] = Bt.T
        sd[L % 0 + "self_attn.o_proj.weight"][:, r] = Bs * (s_o / 2)
    model.load_state_dict(sd)
    hs = _hidden(model, ids)
    with torch.no_grad():
        ln1 = model.model.layers[1].input_layernorm(hs[1])
        nu = ln1.reshape(-1, D).mean(0); nu_hat = nu / nu.norm()
        pc = ln1 @ nu_hat; m_c, s_c = pc.mean().item(), pc.std().item()
    theta = 10000.0 ** (-torch.arange(32) / 32)
    u = torch.zeros(HD); u[:8] = a_prev
    ku = torch.zeros(HD); ku[:8] = a_prev * torch.cos(theta[:8]); ku[32:40] = a_prev * torch.sin(theta[:8])
    for h in (0, 1):
        r = slice(h * HD, (h + 1) * HD)
        sd[L % 1 + "self_attn.q_proj.weight"][r] = u[:, None] * nu_hat[None, :] / m_c
        sd[L % 1 + "self_attn.k_proj.weight"][r] = ku[:, None] * nu_hat[None, :] / m_c
        sd[L % 1 + "self_attn.v_proj.weight"][r] = Bs.T
        sd[L % 1 + "self_attn.o_proj.weight"][:, r] = Bp * (s_o / 2)
    model.load_state_dict(sd)
    hs = _hidden(model, ids)
    with torch.no_grad():
        lnI = model.model.layers[5].input_layernorm(hs[5])
        ratio = (lnI @ Bs).norm(dim=-1).mean() / ((lnI @ Bp).norm(dim=-1).mean() + 1e-9)
    for h in (0, 1):
        sd[L % 1 + "self_attn.o_proj.weight"][:, h * HD:(h + 1) * HD] *= ratio
    model.load_state_dict(sd)
    hs = _hidden(model, ids)
    with torch.no_grad():
        lnI = model.model.layers[5].input_layernorm(hs[5]); qc = lnI @ Bs; kc = lnI @ Bp
        Mq = qc.reshape(-1, 64); Mk = kc.reshape(-1, 64); M = Mq.T @ Mq / len(Mq) + Mk.T @ Mk / len(Mk)
        ev, U_all = torch.linalg.eigh(M)
        Uc = U_all[:, :2 * len(list(slow_pairs))]
        dots = ((qc[:, :-1] @ Uc) * (kc[:, 1:] @ Uc)).sum(-1)
    scale = math.sqrt(target_logit * math.sqrt(HD) / max(dots.mean().item(), 1e-6))
    Wq = torch.zeros(HD, D); Wk = torch.zeros(HD, D)
    for i, p in enumerate(slow_pairs):
        Wq[p] = Bs @ Uc[:, 2 * i] * scale; Wq[p + 32] = Bs @ Uc[:, 2 * i + 1] * scale
        Wk[p] = Bp @ Uc[:, 2 * i] * scale; Wk[p + 32] = Bp @ Uc[:, 2 * i + 1] * scale
    r = slice(0, HD)
    sd[L % 5 + "self_attn.q_proj.weight"][r] = Wq; sd[L % 5 + "self_attn.k_proj.weight"][r] = Wk
    sd[L % 5 + "self_attn.v_proj.weight"][r] = Bs.T
    sd[L % 5 + "self_attn.o_proj.weight"][:, r] = Bo * s_oi
    model.load_state_dict(sd)
    if verbose:
        print("  carrier SNR at L1: m_c %.2f +- %.0f%% (SNR %.1f)  prev rebalance x%.2f  comparator scale %.2f"
              % (m_c, 100 * s_c / m_c, m_c / s_c, ratio, scale))
    return model


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

BUILDERS = {"2b": build2b, "3": build3}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="path to the construction spec JSON")
    ap.add_argument("--output", required=True)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--flag", default=None)
    a = ap.parse_args()

    import json, os
    import transformers

    spec = json.load(open(a.spec))
    version = spec.get("version", "2b")

    # Build the fresh host here, as graft.py does.
    cfg = transformers.AutoConfig.from_pretrained(spec["arch"])
    torch.manual_seed(a.random_seed)
    transformers.set_seed(a.random_seed)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    model.eval()

    kwargs = {k: v for k, v in spec.items()
              if k not in ("arch", "version", "name", "reference")}
    if spec.get("reference"):
        # Which model to mirror is a config choice.
        kwargs["cos_pool"] = empirical_cos(spec["reference"])
    model, info = BUILDERS[version](model, seed=a.random_seed, **kwargs)

    os.makedirs(a.output, exist_ok=True)
    model.save_pretrained(a.output)

    # The same gate the grafts report.
    gate = transformers.AutoModelForCausalLM.from_pretrained(a.output)
    gate.set_attn_implementation("eager")
    gate.eval()
    sc = induction_score(gate, cfg.num_hidden_layers, cfg.num_attention_heads,
                         cfg.vocab_size)
    report = {"spec": spec, "version": version, "random_seed": a.random_seed,
              "induction_max": float(sc.max()),
              "n_heads_gt_0.2": int((sc > 0.2).sum()),
              "build_info": str(info)}
    with open(os.path.join(a.output, "synthetic_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[synthetic] v%s -> %s | induction_max %.4f, heads>0.2: %d"
          % (version, a.output, report["induction_max"], report["n_heads_gt_0.2"]),
          flush=True)

    if a.flag:
        os.makedirs(os.path.dirname(os.path.abspath(a.flag)), exist_ok=True)
        with open(a.flag, "w") as f:
            f.write("synthetic circuit built\n")


if __name__ == "__main__":
    main()
