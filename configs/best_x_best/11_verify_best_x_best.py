"""Is the own-dev-ppl pin also best for transfer?  Every searched Stage-A
rate at 30M into the pinned Stage B."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config)

RANDOM_SEEDS = [42, 43, 44]
VERIFY_BUDGET = 30_000_000
LR_LADDER = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3]
STAGE_B_LR_30M = 5e-4          # the cross's Stage-B pin at 30M (10_best_x_best.py)
WALLTIME_30M = "00:45:00"

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


def deu_a(prefix):
    return dataset_config(
        "language", language="deu_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainA", tokenizer_split="trainA", train_prefix=prefix)


def tur_a(prefix):
    return dataset_config(
        "language", language="tur_Latn", words=1_500_000_000,
        train_words=1_200_000_000, decay_words=100_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="train", tokenizer_split="train", train_prefix=prefix)


def jpn_a(prefix):
    return dataset_config(
        "language", language="jpn_Jpan", words=500_000_000,
        train_words=300_000_000, dev_words=100_000_000, test_words=100_000_000,
        train_split="train", tokenizer_split="train", train_prefix=prefix)


def dyck_a(prefix):
    return dataset_config(
        "shuffle_dyck", num_symbols=128, target_length=2048, p=0.5,
        words=110_000_000, train_words=104_000_000, dev_words=5_000_000,
        test_words=1_000_000, tokenizer_split="train", train_prefix=prefix)


def pos_tag_a(prefix):
    return dataset_config(
        "pos_tag_text", tag_source="ptb", split_map={"train": "trainA", "dev": "dev"},
        source=dataset_config(
            "language", language="eng_Latn", words=4_300_000_000,
            trainA_words=2_000_000_000, trainB_words=2_000_000_000,
            dev_words=100_000_000, test_words=100_000_000),
        words=2_100_000_000, train_words=2_000_000_000, dev_words=100_000_000,
        tokenizer_split="train", train_prefix=prefix)


PRIORS = [eng_a, deu_a, tur_a, jpn_a, pos_tag_a, dyck_a]

# --- model and recipe ---------------------------------------------------------

MODEL_INIT = model_config(architecture_like="config/300M_infi.json", tokenizer="train",
                          vocab_size=16_000, seeded_init=True)
MODEL_PRETRAINED = model_config(use_pretrained=True, vocab_size=16_000, tokenizer="train")
RESET = transition_config("reset_embeddings")
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)


def training(lr, total_dev, slurm_time):
    return training_config(
        total_train="all", total_dev=total_dev, slurm_time=slurm_time,
        sequence_length=2048, batch_size=8, gradient_accumulation_steps=1,
        gpus="0", report_to="wandb", lr=lr, wd=0.1, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0)


globals().update(locals())

# --- the blocks ---------------------------------------------------------------

STAGE_A = [
    {"model": [MODEL_INIT],
     "transition": [RESET],
     "dataset": [prior(VERIFY_BUDGET)],
     "training": [training(lr, total_dev=2_000_000, slurm_time=WALLTIME_30M)
                  for lr in LR_LADDER],
     "scheduler": [SCHEDULER],
     "evaluation": [eval_config("perplexity", split="dev")]}
    for prior in PRIORS]

STAGE_B = [
    {"model": [MODEL_PRETRAINED],
     "transition": [RESET],
     "dataset": [eng_b(VERIFY_BUDGET)],
     "training": [training(STAGE_B_LR_30M, total_dev=5_000_000, slurm_time=WALLTIME_30M)],
     "scheduler": [SCHEDULER],
     "evaluation": [eval_config("lm_harness", tasks="blimp"),
                    eval_config("perplexity", split="dev")]},
]

EXPERIMENTS = {
    "verify_best_x_best": {"stage_A": STAGE_A, "stage_B": STAGE_B},
}
