"""EMA: constant-rate Stage B over 1B English words with weight averages
at three timescales, every prior vs baseline."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config)

RANDOM_SEEDS = [42]

TARGET_WORDS = 1_000_000_000
LR = 5e-4
WARMUP_STEPS = 1000
EMA_TAUS_STEPS = [1_000, 10_000, 100_000]     # decay 0.999, 0.9999, 0.99999
EVAL_EVERY_STEPS = 2000
EMA_TRACK_TOKENS = 1_000_000
RESCALE_STD = 0.02
SMOL = "HuggingFaceTB/SmolLM2-360M"
WALLTIME_A = {30_000_000: "00:45:00", 100_000_000: "01:30:00", 1_000_000_000: "07:00:00"}
WALLTIME_INFI = "24:00:00"
WALLTIME_SMOL = "36:00:00"

# Stage-A pins: the cross's (best_x_best/10_best_x_best.py) for 30M and 100M;
# the 1B-word English pin is the 300M / 1B extension's (2026-07-18).
STAGE_A_LR = {
    ("eng", 30_000_000): 5e-4, ("eng", 100_000_000): 5e-4, ("eng", 1_000_000_000): 5e-4,
    ("dyck", 30_000_000): 5e-4, ("dyck", 100_000_000): 3e-4,
}

# --- corpora: each prior is the first `prefix` words of its corpus ---------

def eng_a(prefix):
    return dataset_config(
        "language", language="eng_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainA", tokenizer_split="trainA", train_prefix=prefix)


def eng_b(prefix):
    return dataset_config(
        "language", language="eng_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainB", tokenizer_split="trainB", train_prefix=prefix)


def dyck_a(prefix):
    return dataset_config(
        "shuffle_dyck", num_symbols=128, target_length=2048, p=0.5,
        words=110_000_000, train_words=104_000_000, dev_words=5_000_000,
        test_words=1_000_000, tokenizer_split="train", train_prefix=prefix)


PRIORS = {"eng": eng_a, "dyck": dyck_a}

# --- models and recipe --------------------------------------------------------

MODEL_INIT = model_config(architecture_like="config/300M_infi.json", tokenizer="train",
                          vocab_size=16_000, seeded_init=True)
MODEL_PRETRAINED = model_config(use_pretrained=True, vocab_size=16_000, tokenizer="train")
SMOL_INIT = model_config(architecture_like=SMOL, tokenizer="train", vocab_size=16_000, seeded_init=True)
SMOL_BODY = model_config(from_hub=SMOL, tokenizer="train", vocab_size=16_000)
RESET = transition_config("reset_embeddings")
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)
CONSTANT_LR = scheduler_config("wsd", warmup_steps=WARMUP_STEPS)


def training(lr, total_dev, slurm_time, grad_accum=1, **extra):
    return training_config(
        total_train="all", total_dev=total_dev, slurm_time=slurm_time,
        sequence_length=2048, batch_size=8, gradient_accumulation_steps=grad_accum,
        gpus="0", report_to="wandb", lr=lr, wd=0.1, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0, **extra)


def ema_training(grad_accum=1, slurm_time=WALLTIME_INFI, **extra):
    t = training(LR, total_dev=5_000_000, slurm_time=slurm_time, grad_accum=grad_accum, **extra)
    t["CONFIG"].update(EMA_TAUS_STEPS=EMA_TAUS_STEPS, EVAL_EVERY_STEPS=EVAL_EVERY_STEPS,
                       PATIENCE_EVALS=10 ** 9, EMA_TRACK_TOKENS=EMA_TRACK_TOKENS)
    return t


# No Stage A: the fresh model copied through untrained.
NO_STAGE_A = {
    "model": [MODEL_INIT],
    "transition": [RESET],
    "dataset": [dataset_config("none", words=2_300_000_000, train_words=2_000_000_000,
                               dev_words=100_000_000, test_words=100_000_000)],
    "training": [training_config(   # placeholder; never runs
        checkpoint_interval=[8_000_000], eval_interval=[], total_train=8_000_000,
        total_dev=5_000_000, sequence_length=2048, batch_size=8,
        gradient_accumulation_steps=1, gpus="0", lr=5e-4, wd=0.1, fp16=False, bf16=True)],
    "scheduler": [scheduler_config("warmup_cosine", warmup_steps=100, min_lr_rate=0.1)],
    "evaluation": [],
}

globals().update(locals())

# --- Stage A: the priors at their pins, and the SmolLM2 hosts -----------------

PRIOR = {
    (arm, budget): {
        "model": [MODEL_INIT],
        "transition": [RESET],
        "dataset": [PRIORS[arm](budget)],
        "training": [training(STAGE_A_LR[arm, budget], total_dev=2_000_000,
                              slurm_time=WALLTIME_A[budget])],
        "scheduler": [SCHEDULER],
        "evaluation": [eval_config("perplexity", split="dev")]}
    for (arm, budget) in STAGE_A_LR}

SMOL_NONE = {**NO_STAGE_A, "model": [SMOL_INIT]}
SMOL_PRETRAINED = {**NO_STAGE_A, "model": [SMOL_BODY]}

# --- Stage B: one pass over 1B words at a constant rate, with EMAs -----------

ARMS = {
    "plain":         ([NO_STAGE_A, PRIOR["eng", 30_000_000], PRIOR["eng", 100_000_000],
                       PRIOR["eng", 1_000_000_000], PRIOR["dyck", 30_000_000],
                       PRIOR["dyck", 100_000_000]],
                      ema_training()),
    "rescaled":      ([PRIOR["eng", 1_000_000_000]],
                      ema_training(rescale_std=RESCALE_STD)),
    "smol":          ([SMOL_NONE, SMOL_PRETRAINED],
                      ema_training(grad_accum=2, slurm_time=WALLTIME_SMOL)),
    "smol_rescaled": ([SMOL_PRETRAINED],
                      ema_training(grad_accum=2, slurm_time=WALLTIME_SMOL, rescale_std=RESCALE_STD)),
}

globals().update(locals())

EXPERIMENTS = {
    name: {
        "stage_A": stage_a,
        "stage_B": [{"model": [MODEL_PRETRAINED],
                     "transition": [RESET],
                     "dataset": [eng_b(TARGET_WORDS)],
                     "training": [stage_b_training],
                     "scheduler": [CONSTANT_LR],
                     "evaluation": [eval_config("lm_harness", tasks="blimp"),
                                    eval_config("perplexity", split="dev")]}],
    }
    for name, (stage_a, stage_b_training) in ARMS.items()}
