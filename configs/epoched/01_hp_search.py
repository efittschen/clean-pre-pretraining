"""Epoched Stage-B search on the baseline: (epochs, lr, wd) at 1M and 10M.
Produces the pins 10_* uses."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config)

RANDOM_SEEDS = [42, 43, 44]

EPOCHED_SIZES = [1_000_000, 10_000_000]
EPOCHED_EPOCHS = [4, 8]
EPOCHED_LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
EPOCHED_WDS = [0.1, 1.0, 2.0, 5.0]
EPOCHED_WALLTIME = {1_000_000: "00:30:00", 10_000_000: "02:00:00", 100_000_000: "08:00:00"}

# --- corpus: the first `prefix` words of English trainB -----------------------

def eng_b(prefix):
    return dataset_config(
        "language", language="eng_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainB", tokenizer_split="trainB", train_prefix=prefix)


# --- model and recipe ---------------------------------------------------------

MODEL_INIT = model_config(architecture_like="config/300M_infi.json", tokenizer="train",
                          vocab_size=16_000, seeded_init=True)
MODEL_PRETRAINED = model_config(use_pretrained=True, vocab_size=16_000, tokenizer="train")
RESET = transition_config("reset_embeddings")
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)


def training(lr, wd, epochs, slurm_time):
    return training_config(
        total_train="all", total_dev=5_000_000, slurm_time=slurm_time,
        sequence_length=2048, batch_size=8, gradient_accumulation_steps=1,
        gpus="0", report_to="wandb", lr=lr, wd=wd, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0,
        num_train_epochs=epochs)


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

# --- the blocks: one per size, the (wd x epochs x rate) cross inside ----------

STAGE_B = [
    {"model": [MODEL_PRETRAINED],
     "transition": [RESET],
     "dataset": [eng_b(words)],
     "training": [training(lr, wd, epochs, slurm_time=EPOCHED_WALLTIME[words])
                  for wd in EPOCHED_WDS for epochs in EPOCHED_EPOCHS for lr in EPOCHED_LRS],
     "scheduler": [SCHEDULER],
     "evaluation": [eval_config("lm_harness", tasks="blimp"),
                    eval_config("perplexity", split="dev")]}
    for words in EPOCHED_SIZES]

EXPERIMENTS = {
    "epoched_hp_search": {"stage_A": [NO_STAGE_A], "stage_B": STAGE_B},
}
