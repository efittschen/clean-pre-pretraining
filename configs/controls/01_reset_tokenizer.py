"""CONTROL: does reset_embeddings erase token identity?  Stage B on the same
vs a permuted tokenizer; expected null.  BabyLM line (baby-llama-2-345m)."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config,
                            tokenizer_with_pos_tags_and_dyck, tokenizer_permuted)

RANDOM_SEEDS = [42, 43, 44]

# --- tokenizers: the BabyLM line's extended vocab, and the same with ids permuted

EXTENDED_TOKENIZER = tokenizer_with_pos_tags_and_dyck(
    "JLTastet/baby-llama-2-345m", tag_source="ptb", num_dyck_symbols=128)
PERMUTED_TOKENIZER = tokenizer_permuted(EXTENDED_TOKENIZER)   # a different permutation per seed
TOKENIZERS = {"sametok": EXTENDED_TOKENIZER, "permtok": PERMUTED_TOKENIZER}

# --- corpora ------------------------------------------------------------------

BABYLM = dataset_config("babylm", words=10_000_000, train_words=10_000_000, dev_words=0)
LANGUAGE = dataset_config(
    "language", language="eng_Latn", words=4_300_000_000,
    trainA_words=2_000_000_000, trainB_words=2_000_000_000,
    dev_words=100_000_000, test_words=100_000_000, train_split="trainA")
POS_TAG = dataset_config("pos_tag", source=BABYLM, tag_source="ptb")
NONE = dataset_config("none", words=2_300_000_000, train_words=2_000_000_000,
                      dev_words=100_000_000, test_words=100_000_000)

# --- model and recipe ---------------------------------------------------------

BABY_MODEL = model_config(architecture_like="JLTastet/baby-llama-2-345m", tokenizer=EXTENDED_TOKENIZER)
RESET = transition_config("reset_embeddings")

STAGE_A_TRAINING = training_config(     # 8M tokens at the line's pinned cell
    checkpoint_interval=[8_000_000], eval_interval=[], total_train=8_000_000, total_dev=5_000_000,
    sequence_length=128, batch_size=128, gradient_accumulation_steps=2, gpus="0", report_to="wandb",
    lr=5e-4, wd=0.1, fp16=True, bf16=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6,
    max_grad_norm=1.0, slurm_time="03:00:00", slurm_memory="16GB")

EPOCH_TOKENS = 14_794_752               # 14.8M rounded down to whole steps of 128 x 128
EVAL_EVERY = 2 * EPOCH_TOKENS // 10     # 10 dev evals across the 2 epochs
STAGE_B_TRAINING = training_config(     # the baby-llama-2 recipe, 2 epochs
    checkpoint_interval=[EPOCH_TOKENS, 2 * EPOCH_TOKENS],
    eval_interval=list(range(EVAL_EVERY, 10 * EVAL_EVERY + 1, EVAL_EVERY)),
    total_train=14_800_000, total_dev=2_000_000, num_train_epochs=2,
    sequence_length=128, batch_size=128, gradient_accumulation_steps=2, gpus="0", report_to="wandb",
    lr=7e-4, wd=5.0, fp16=True, bf16=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6,
    max_grad_norm=1.0, slurm_time="02:00:00", slurm_memory="24GB")

# --- Stage A arms -------------------------------------------------------------

STAGE_A = {
    "baseline": {"model": [BABY_MODEL],
                 "transition": [RESET],
                 "dataset": [NONE],
                 "training": [STAGE_A_TRAINING],    # placeholder; never runs
                 "scheduler": [scheduler_config("warmup_cosine", warmup_steps=100, min_lr_rate=0.1)],
                 "evaluation": []},
    "language": {"model": [BABY_MODEL],
                 "transition": [RESET],
                 "dataset": [LANGUAGE],
                 "training": [STAGE_A_TRAINING],
                 "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=1.0, min_lr_rate=0.1)],
                 "evaluation": []},
    "pos_tag":  {"model": [BABY_MODEL],
                 "transition": [RESET],
                 "dataset": [POS_TAG],
                 "training": [STAGE_A_TRAINING],
                 "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=1.0, min_lr_rate=0.1)],
                 "evaluation": []},
}

globals().update(locals())

# --- Stage B: plain BabyLM, once per tokenizer condition ----------------------

EXPERIMENTS = {
    f"{arm}_{condition}": {
        "stage_A": [STAGE_A[arm]],
        "stage_B": [{"model": [model_config(use_pretrained=True, tokenizer=tokenizer)],
                     "transition": [RESET],
                     "dataset": [BABYLM],
                     "training": [STAGE_B_TRAINING],
                     "scheduler": [scheduler_config("warmup_cosine", warmup_steps=600, min_lr_rate=0)],
                     "evaluation": [
                         eval_config("lm_harness", tasks="blimp",
                                     slurm={"STEAMROLLER_TIME": "02:00:00", "STEAMROLLER_MEMORY": "24GB"}),
                         eval_config("perplexity", split="dev",
                                     slurm={"STEAMROLLER_TIME": "01:00:00", "STEAMROLLER_MEMORY": "24GB"})]}],
    }
    for arm in STAGE_A for condition, tokenizer in TOKENIZERS.items()}
