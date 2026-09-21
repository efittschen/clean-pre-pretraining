# Induction-head transplant

Does a pre-pretraining prior have to be a whole trained body to help, or is
it enough to install its induction heads into a random model?

## The donor

`donor/` is the **language 100M** Stage A prior. Its induction (copy) score
peaks at **0.7089 on L5H8**; four heads exceed 0.2:

| head | copy score |
|---|---|
| L5H8 | 0.7089 |
| L7H1 | 0.5903 |
| L4H2 | 0.5893 |
| L8H6 | 0.2888 |

Measured with uniform-random token ids on a repeated sequence `[core, core]`
(period N=128), so the probe sees attention *geometry*, not content — the part
that can survive an embedding reset.

## The variants

| variant | copied from the donor | asks |
|---|---|---|
| `baseline` | nothing — seeded random init | reference floor |
| `induction4` | the 4 heads above | **the hypothesis** |
| `induction_top1` | only L5H8 | is one head enough? |
| `random4` | L5H6 L7H7 L4H0 L8H4 | **control** |
| `full_prior` | the entire transformer body | reference ceiling |

`random4` takes four *different* heads from the *same four layers*, so layer
depth is held fixed and only head identity varies. Without it, a positive
`induction4` result cannot be distinguished from "any four trained heads help".

`full_prior` copies every body tensor but leaves embeddings and the LM head
random — exactly the `reset_embeddings` transition the main pipeline uses, so
it reproduces the published prior arm inside this same setup.

All five variants are built here and run through the same Stage B, so every
comparison is internal — no need to match numbers to a historical pipeline.

## What a head is

The architecture is Llama-style, 12 layers x 12 heads, hidden 768, head_dim 64,
with `num_key_value_heads == num_attention_heads` (no GQA), so head H of layer L
is a clean slice:

```
q_proj.weight[H*64:(H+1)*64, :]     rows
k_proj.weight[H*64:(H+1)*64, :]     rows
v_proj.weight[H*64:(H+1)*64, :]     rows
o_proj.weight[:, H*64:(H+1)*64]     columns
```

Everything else — embeddings, LM head, MLPs, norms, all other heads — stays at
the recipient's random init. `--with-layernorm` also copies `input_layernorm`
for any donating layer.

## Step 1 — build the variants and RUN THE GATE FIRST

```bash
python scripts/make_transplants.py --score
```

A head's behaviour depends on the representations feeding it. Dropped into a
random-init residual stream, a transplanted head may not be an induction head
any more. `--score` re-measures the copy score on every variant.

**If `induction4` is not clearly above `baseline`, the premise has failed —**
**stop here.** That check costs minutes; Stage B costs GPU-days.

## Step 2 — Stage B

Only if the gate passes. Run every variant x seed through the same Stage B.
Targets included in this export: 10M, 30M, 100M.

| setting | value |
|---|---|
| `total_train` | all (single pass over every window) |
| `sequence_length` | 2048 |
| `batch_size` | 8 (x grad_accum 1 = 16,384 tokens/step) |
| `precision` | bf16 |
| `optimizer` | AdamW, beta 0.9/0.999, eps 1e-6, max_grad_norm 1.0 |
| `weight_decay` | 0.1 |
| `scheduler` | cosine_with_min_lr, warmup_ratio 0.1, min_lr_rate 0.1 |
| `total_dev` | 5,000,000 |

| Stage B target | LR |
|---|---|
| 10M words | 0.0005 |
| 30M words | 0.0005 |
| 100M words | 0.0005 |

```bash
python scripts/model_train_words.py \
    --model models/<variant>/seed<N> \
    --train stage_b_data/<target>Mw/seed<N>/trainB.tok \
    --dev   stage_b_data/<target>Mw/seed<N>/dev.tok \
    --tokenizer tokenizers/<target>Mw/seed<N> \
    --output out/<variant>_B<target>M_seed<N>
```

Check `--help` for exact flag names on your cluster; the above is the shape.
The variants already have random embeddings, so do **not** apply
`model_reset_embeddings.py` again before Stage B — it ships only because the
reference pipeline uses it between stages.

## How to read the result

For each Stage B target, compare dev perplexity across variants:

```
baseline  >  random4  >  induction4  >  full_prior      <- heads carry the benefit
baseline  ~  random4  ~  induction4  <  full_prior      <- the body carries it
baseline  ~  induction4 ~ random4 ~ full_prior          <- nothing transferred here
```

For orientation, the published cross measured these no-prior baselines with
this recipe (median over seeds 42/43/44):

| Stage B target | baseline dev ppl |
|---|---|
| 10M | 266.1 |
| 30M | 115.5 |
| 100M | 49.5 |

Those come from a 16000-vocab tokenizer; the donor is 16001. Only embeddings
differ and they are random in every variant here, so the internal comparison is
unaffected — but treat the numbers above as orientation, not as a fifth arm.

## Caveat worth stating in any writeup

Induction is not the whole transfer story. In the same measurement `dyck` scores
0.011-0.012 and `pos` 0.089-0.192 — essentially no induction — yet both help in
the published cross. So whatever this experiment shows, the dyck benefit is
coming from something other than induction heads.
