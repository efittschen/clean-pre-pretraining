# Pre-pretraining

Does training a randomly-initialised language model on a **structural** corpus
first — bracket languages, POS-tag streams, another natural language — then
resetting its embeddings and training on the real corpus, beat training on the
real corpus alone?

Two stages, and the transition between them is the point:

```
 Stage A                    transition                Stage B
 ───────                    ──────────                ───────
 train on a structural  →   reset_embeddings      →   train on the target
 prior (Dyck, POS tags,     input embeddings +        corpus (English)
 German, Japanese, ...)     LM head redrawn from
                            N(0, 0.02²); the
                            transformer BODY
                            carries over
```

Resetting the embeddings is what makes the comparison meaningful: whatever
transfers cannot be vocabulary, because the vocabulary is thrown away. Only the
body survives.

Every reported number is a delta against the `none` arm — no Stage A, a fresh
model copied straight through — with everything else held fixed.

**This repository is code only.** No weights, no corpora, no run outputs.
Everything regenerates from here.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. edit ONE file for your cluster: accounts, partitions, batch budgets
$EDITOR site_config.py

# 2. see what would run, without running it
scons EXPERIMENT=configs/demo/00_smoke.py -n

# 3. run it
scons EXPERIMENT=configs/demo/00_smoke.py
```

`-n` prints the exact command for every job and submits nothing. Use it first,
always — it is the cheapest way to see what a config expands to.

## Start here: the two demos

| | What it shows | Cost |
|---|---|---|
| `configs/demo/00_smoke.py` | The graph runs end to end. Dyck → Dyck, no corpus download. **Not a result** — both stages are the same synthetic data. | minutes, 1 GPU |
| `configs/demo/01_prior_vs_baseline.py` | The real question at 1/1000 scale: Dyck prior vs no prior, both → English, compared on dev perplexity. | ~1 hour, 1 GPU |
| `configs/demo/02_pos_ramp.py` | The POS ramp at 1/100 scale: tags → words handed off gradually, after a tag-only Stage A vs from scratch vs plain training, on one perplexity axis. | ~1 hour, 1 GPU |

Demo 2 is the shape of every experiment in `configs/best_x_best/`. Read its
docstring before the numbers — it is one seed at a scale far below where the
effect was characterised, so it illustrates the method, not the finding.

## The pipeline

One line of research: **prior transfer** (`configs/best_x_best/` and the
units built on it). Does a *structural* Stage-A prior leave a model better at
English? Stage A trains on Dyck brackets, POS-tag streams or another language;
the embeddings are reset; Stage B trains on English. Measured against the
`none` arm. One model line: 300M_infi, one 16k tokenizer per corpus and size,
word budgets, single pass. Every other unit (`epoched/`, `rescaling/`, `ema/`,
`pos_ramp`) builds on the same corpora and carries its own copy of the pins.

The original trees' second line, the *secrets* experiments (can a learned
secret survive the reset?), is not ported here. `docs/INVENTORY.md` records
where it lives and what state it is in.

## The experiment catalogue

Run any of these with `scons EXPERIMENT=<path>`. Ordering within a group is
a dependency order, not a preference: the search config produces the learning
rates that the final config hard-codes.

### configs/best_x_best/ — the main line: search, cross, verification

300M_infi (127M params at 16k vocab), one tokenizer per corpus and size, word
budgets {1, 3, 10, 30, 100}M, single pass. Six priors: English, German,
Turkish and Japanese text, the PTB tag stream of English, and shuffled Dyck.

| Config | Question |
|---|---|
| `01_hp_search` | **Run first.** Stage-A LR ladder for every prior × budget, and the Stage-B LR ladder on the baseline. Produces the pins. |
| `10_best_x_best` | **The final cross**: 6 priors × 5 × 5 at the pinned LRs, plus the baseline row. |
| `11_verify_best_x_best` | Is the rate chosen on a prior's own dev perplexity also the best for transfer? Every searched Stage-A rate at 30M into the pinned 30M Stage B. |

Every config is self-contained: it declares its own corpora, model, recipe,
stage blocks and pins, and imports only `experiment_dsl` (and `site_config`
where it sizes a batch). `10_best_x_best` holds the reference copy of the
pins; the configs that reuse them carry their own copy, so change all copies
together.

### configs/epoched/ — the same cross with Stage B trained to convergence

| Config | Question |
|---|---|
| `01_hp_search` | **Run first.** Epochs × lr × wd on the baseline at 1M and 10M words. Produces the epoched pins. |
| `10_best_x_best` | Every prior at its pinned rate into Stage B at 4 and 8 epochs. Does the prior survive repetition? |

Carries a copy of the cross's pins; the Stage-A declarations are identical, so the
Stage-A runs are shared with the single-pass cross.

### configs/pos_ramp/ — hand off from tags to words gradually instead of resetting

| Config | Question |
|---|---|
| `01_hp_search` | **Run first.** The pure-POS Stage A (posA) and the ramped Stage B from a fresh init (rampB), each over the LR ladder at every size. Produces the pins. |
| `10_best` | posA at its pin into rampB at its pin, embeddings carried over, per size; against the same ramp from scratch and the cross baseline. |

The ramp emits each word either as its natural tokens or as its tag token,
with the word probability rising from 0 to 1 over the first 10 percent of
Stage B. The tag tokens extend the cross's own per-size tokenizer and carry a
private-use marker so they are inert on natural text, which makes the word
side byte-identical to the baseline's tokenization; the aligned builder
asserts it. That is what lets these perplexities sit on the cross's axis.
Not a reset arm: the ramp starts from the tag embeddings.

### configs/rescaling/ — does rescaling the transferred body to init scale change the result?

| Config | Question |
|---|---|
| `10_best_x_best` | The cross again, with every body matrix rewritten as `W = c·W'` at init scale before Stage B. Do the cells where a small prior hurts a large target go away? |

Same priors and pins as `best_x_best/10_best_x_best.py`, copied in. Two
more rescaling tests, a 1B-word prior and SmolLM2's pretrained body under a
long constant-LR Stage B, are still only in the original tree's `long_stageB/`.

### configs/controls/ — checks on the method itself

| Config | Question |
|---|---|
| `01_reset_tokenizer` | Does `reset_embeddings` really erase token identity? Stage B twice from one Stage A, with the tokenizer's ids permuted the second time. A null is the desired result. BabyLM line, two epochs. |
| `02_hu_2x2` | Does the shuffle-Dyck → C4 result of Hu et al. 2025 replicate in our code, under the hyperparameters in Hu's code and under those in the paper? Pythia-160m. Reading: `docs/REPRODUCTION_2x2.md`. |
| `03_ngram_null` | Is a structural prior's gain structural, or would any warm start do? A Stage A of unigram-sampled English words, real lexicon and no syntax. Still on the discarded BabyLM recipe, like `01_reset_tokenizer`; both need re-expressing on the 300M line to stay. |

### configs/frozen_body/ — what can a fresh embedding alone do on a frozen pretrained body?

| Config | Question |
|---|---|
| `01_hp_search` | SmolLM2-360M's body frozen, a fresh tied 16k embedding drawn at the donor's own RMS, trained on 100M words of English: which embedding learning rate, and does the final norm need to train? Produces the pin. |

The historical half is not here yet: the same host on a 45M-word 1750–1910
corpus at the pinned rate, 10 epochs with a checkpoint per epoch and EMA, a
re-tokenized variant that separates domain from tokenizer, and the
contamination probes (historical cloze by citation year, anachronism minimal
pairs, modern-text scoring, embedding lock-in). The graph has no builder for
that corpus. The inputs exist on this cluster: the five cleaned period files
`scratch_clean_data{1750,1820,_1850,_1880,_1910}.train` in the original
transplant export (about 10M words each), the raw ones beside the Skipjack
repo, and `hist_cloze.parquet`. The builder has to concatenate the periods,
hold out the last 1M words of each as dev, and train the corpus's own 16k
tokenizer on it so no modern vocabulary leaks in.

### configs/ema/ — constant learning rate with weight averages, to 1.4B tokens

| Config | Question |
|---|---|
| `10_long_stage_b` | Every prior, plus SmolLM2's body, into one pass over 1B words at a constant rate with EMAs at three timescales. How does a prior's advantage evolve without a decaying schedule, and how far ahead of the raw weights is their average? |

Priors at the cross's pins, copied in. Ten arms, all complete
in the original tree at seed 42 (`work/long_stageB/`). Two more EMA runs,
Qwen2.5-0.5B transferred and from scratch on the historical corpus, wait on
the same missing builder.

### configs/transplant/ — can a circuit be copied instead of trained?

| Config | Question |
|---|---|
| `70_induction_transplant` | The core 5 arms. **Its construction gate failed** — read the header. |
| `71_synthetic_circuits` | Build the circuit by hand instead of copying it. |

The original export's follow-ups (bias vs weights, public donor models, the
other prior families) are not ported; `docs/INVENTORY.md` lists them.

### Read before trusting the transplant numbers

`scripts/transplant/` holds the graft and synthetic-circuit builders the two
configs above drive. **Read `docs/INVENTORY.md` first**: the core transplant's
gate FAILED (grafted `induction_max` 0.0061 vs baseline 0.0059) and Stage B
was run regardless. The subprojects the original trees carried (plasticity,
procedural pretraining, NCA, the BabyLM evaluation pipeline) are not here.

## How an experiment is described

One file per experiment, under `configs/`. A config is **declarative data**,
not a program — SCons reads it via `Variables()`, and running it directly does
nothing.

```python
EXPERIMENTS = {
    "my_experiment": {
        "stage_A": [ {model: [...], dataset: [...], training: [...], ...} ],
        "stage_B": [ {...} ],
    },
}
RANDOM_SEEDS = [42, 43, 44]
```

Each key maps to a **list**, and the graph takes the cartesian product — so one
block can expand to a whole grid. Every Stage-A output seeds every Stage-B
block, and the entire graph is rebuilt per seed.

The vocabulary (`model_config`, `dataset_config`, `training_config`,
`scheduler_config`, `transition_config`, `eval_config`) lives in
`experiment_dsl.py`. The optimizer is AdamW and its knobs are plain
`training_config` keywords (`lr`, `wd`, `adam_epsilon`, ...); there is no
separate optimizer block.

## Hyperparameters live in the path

A run's directory *is* its configuration:

```
work/models/model_like_pythia-160m-transition/
  reset_embeddings/<stage_A_dataset>/tok<N>M/bs32/lr5e-04/wd0.1/wr0.1/unfrozen/seed42/
    stage_A/checkpoint-<step>-transition/
      reset_embeddings/<stage_B_dataset>/tok<M>M/.../seed42/
        stage_B/checkpoint-final/
```

Stage B nests *inside* the Stage-A checkpoint that produced it, so the tree
records provenance directly. Every collector in `results/` parses these paths,
which makes `build/paths.py` a naming contract: change it and you rename every
run and orphan every existing result.

## Run order is not cosmetic

The HP searches **produce** the learning rates the cross configs **hard-code**.
Selection is outcome-blind: each arm's HPs come from that arm's own dev
perplexity, never from Stage B or BLiMP. Running a cross before its search
means running it against unsourced numbers.

```bash
scons EXPERIMENT=configs/best_x_best/01_hp_search.py
# read off the winners, paste them into the pins at the top of 10_best_x_best.py, then:
scons EXPERIMENT=configs/best_x_best/10_best_x_best.py
```

## Layout

```
site_config.py      cluster accounts, partitions, validated batch sizes  ← edit this
SConstruct          the build graph: builders, the pipeline, the main loop
experiment_dsl.py   the vocabulary configs are written in
build/
  paths.py          run-directory naming + the cartesian product
  steamroller.py    SLURM task config
configs/
  demo/             start here
  best_x_best/      the main line: search, cross, verification (300M_infi)
  epoched/          the same cross with Stage B trained to convergence
  rescaling/        the same cross with the transferred body rescaled to init scale
  controls/         the reset-tokenizer control and the Hu et al. reproduction
  frozen_body/      a fresh embedding on a frozen pretrained body
  ema/              constant learning rate with weight averages, to 1.4B tokens
  pos_ramp/         the POS→word ramp on the cross's own tokenizers
scripts/
  common/           what every worker shares: the dataloader, the grammar
                    generators, the POS tag inventory, the word-prefix rule
                    and memmap writer (textio.py), the corpus streamer
                    (corpus.py) and the POS-ramp dataset (pos_transition.py)
  data/             corpus builders
  tokenizer/        train / extend / permute / tokenize
  model/            init, embedding reset, AND train.py — the one trainer
  eval/             perplexity.py — the one evaluator — and lm_harness.py
  transplant/       host builders + the synthetic circuit constructions
results/
  collect/          walk work/ and emit a table
  figures/          turn those tables into figures
  tables/           the collected tables, committed
docs/
```

## One trainer, one evaluator

`scripts/model/train.py` replaces eight near-identical trainers, and
`scripts/eval/perplexity.py` replaces five evaluators. They differed only in
options that were **already in the config JSONs**, so nothing new had to be
invented to merge them:

| Set this in the config | train.py does |
|---|---|
| `total_train="all"` | consume every window; save `checkpoint-final` |
| `rescale_std=0.02` | reparametrise the body as `W = c·W'` at init scale |
| `log_eval_steps=[...]` | dump `log_history.json` |
| `probe_steps=[...]` | per-token dev probe by position and frequency bin |
| `model_load={...}` | forward kwargs to `from_pretrained` (bf16, flash-attn) |
| scheduler `reduce_lr_on_plateau` | train to convergence, keep BEST dev |
| `ema_taus_steps=[...]` | EMA-of-weights tracking with early stopping |
| `freeze_final_norm=True` | also freeze `model.norm` |
| `random_chunk=True` | sample random windows instead of walking the stream |
| a `pos_transition` dataset with `transition_schedule={...}` | stream the aligned tag / word files and ramp `P(word)` on the schedule; dev is scored on both sides |
| *(a `matrix_scales.json` beside the model)* | install fixed per-matrix forward multipliers — no flag needed, models without one are untouched |

The original tree kept these apart because SCons hashes builder-script
contents, so editing a live trainer would have invalidated every cached run.
This tree starts with no build state, so the merge costs nothing.

The induction-transplant export shipped a *second* copy of this engine so it
could run standalone elsewhere. In one repository that no longer buys anything,
so it is gone: the last four differences became the last four rows above. The
POS-ramp trainer was a third copy, ~150 shared lines around one dataset class;
that class now lives in `scripts/common/pos_transition.py` and the trainer
selects it from the dataset config.

The same rule was applied to every script that existed only as a one-flag
variant of another: the word-capped tokenizer (`tokenize_file.py --words`), the
seeded model init (`like.py --random_seed`), and the MeCab-segmenting Japanese
downloader (inside `fw_language.py`). The three corpus downloaders share the
stream / count / shuffle loop in `scripts/common/corpus.py`, and the
word-prefix rule that keeps the tokenizer trainer, the tokenizer and the
aligned tagger on exactly the same text is one function in
`scripts/common/textio.py`.

The original tree's `optimizer_config` block is gone. Every run wrote it to
disk and hashed it, and no script ever read it; the optimizer was always
AdamW with the `lr`, `wd`, betas and `eps` from `training_config`. Now that is
the only place they live.

## Before you trust a number

- `docs/INVENTORY.md` — every experiment in both original repositories, what
  state it is in, and which are superseded or abandoned.
- `docs/PROVENANCE.md` — dataset revision SHAs, the eval-harness commit, the
  environment that produced the runs.
- `docs/CLUSTER_NOTES.md` — why specific nodes are excluded. Each ban is on
  measured failures; read it before removing one.
