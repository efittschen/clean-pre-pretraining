"""Frozen body: SmolLM2-360M's body frozen, a fresh tied embedding trained
on 100M words; the embedding LR search."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config)

RANDOM_SEEDS = [42]

DONOR = "HuggingFaceTB/SmolLM2-360M"
EMBED_STD = 0.13445          # the donor's trained embedding RMS
BUDGET = 100_000_000
WALLTIME = "01:30:00"
LR_LADDER_EMB = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
NORM_ARMS = {"normtrain": False, "normfroz": True}    # arm -> FREEZE_FINAL_NORM

# --- corpus: the first `prefix` words of English trainB -----------------------

def eng_b(prefix):
    return dataset_config(
        "language", language="eng_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainB", tokenizer_split="trainB", train_prefix=prefix)


# --- models and recipe --------------------------------------------------------

DONOR_BODY = model_config(from_hub=DONOR, tokenizer="train", vocab_size=16_000)
MODEL_PRETRAINED = model_config(use_pretrained=True, vocab_size=16_000, tokenizer="train")
RESET_AT_DONOR_RMS = transition_config("reset_embeddings", std=EMBED_STD)
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)


def training(lr, freeze_final_norm):
    return training_config(
        total_train="all", total_dev=5_000_000, slurm_time=WALLTIME,
        sequence_length=2048, batch_size=8, gradient_accumulation_steps=1,
        gpus="0", report_to="wandb", lr=lr, wd=0.0, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0,
        freeze_layers=True, unfrozen_top=0, unfrozen_bottom=0,
        freeze_final_norm=freeze_final_norm)


# Stage A: the donor, untrained, copied through.
FROZEN_HOST = {
    "model": [DONOR_BODY],
    "transition": [transition_config("reset_embeddings")],
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

# --- Stage B: the embedding only, one arm with the final norm trainable -------

STAGE_B = [
    {"model": [MODEL_PRETRAINED],
     "transition": [RESET_AT_DONOR_RMS],
     "dataset": [eng_b(BUDGET)],
     "training": [training(lr, freeze_final_norm) for lr in LR_LADDER_EMB],
     "scheduler": [SCHEDULER],
     "evaluation": [eval_config("perplexity", split="dev")]}
    for arm, freeze_final_norm in NORM_ARMS.items()]

EXPERIMENTS = {
    "frozen_body_lr_search": {"stage_A": [FROZEN_HOST], "stage_B": STAGE_B},
}
