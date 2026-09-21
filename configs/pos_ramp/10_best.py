"""POS ramp final runs: posA into rampB at the pinned rates, vs
ramp-from-scratch and the cross baseline.  Stream, tokenizer and blocks are
declared as in 01_hp_search.py, so the fresh-init ramp is a cache hit."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config,
                            tokenizer_with_pos_tags)

RANDOM_SEEDS = [42, 43, 44]

SIZES = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000]
WALLTIME = {1_000_000: "00:20:00", 3_000_000: "00:20:00", 10_000_000: "00:30:00",
            30_000_000: "00:45:00", 100_000_000: "01:30:00"}
TAG_SOURCE = "ptb"
WORDS_PER_DATAPOINT = 1700      # stride in source words between examples (~1.07 tokens/word x 2048)
DEV_PREFIX = 5_000_000          # words of dev to align: ~7M word tokens, above the 5M-token eval
RAMP_FRACTION = 0.10            # the ramp window as a fraction of the Stage-B words

PINS_POS = {1_000_000: 1e-3, 3_000_000: 1e-3, 10_000_000: 3e-4,
            30_000_000: 3e-4, 100_000_000: 3e-4}      # 100M provisional
PINS_RAMP = {1_000_000: 3e-3, 3_000_000: 2e-3, 10_000_000: 1e-3,
             30_000_000: 7e-4, 100_000_000: 7e-4}

# --- corpus: the aligned tag / word stream of the first `size` words of trainB -

def tag_stream(size):
    return dataset_config(
        "pos_transition", language="eng_Latn", tag_source=TAG_SOURCE,
        words=4_300_000_000, trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainB", tokenizer_split="trainB",
        train_prefix=size, dev_prefix=DEV_PREFIX,
        words_per_datapoint=WORDS_PER_DATAPOINT, aligned=True)


POS_ONLY = {"shape": "linear", "warmup_words": 0, "transition_words": 0,
            "start_prob": 0.0, "end_prob": 0.0}


def ramp(size):
    return {"shape": "linear", "warmup_words": 0,
            "transition_words": int(RAMP_FRACTION * size), "start_prob": 0.0, "end_prob": 1.0}


# --- models and recipe: the cross's per-size tokenizer plus marker-prefixed tags

EXT_TOKENIZER = tokenizer_with_pos_tags("train", tag_source=TAG_SOURCE, marker=True)
MODEL_INIT = model_config(architecture_like="config/300M_infi.json", tokenizer="train",
                          vocab_size=16_000, seeded_init=True)
MODEL_INIT_EXT = model_config(architecture_like="config/300M_infi.json", tokenizer=EXT_TOKENIZER,
                              vocab_size=16_000, seeded_init=True)
MODEL_PRETRAINED_EXT = model_config(use_pretrained=True, tokenizer=EXT_TOKENIZER, vocab_size=16_000)
RESET = transition_config("reset_embeddings")
KEEP_EMBEDDINGS = transition_config("none")
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)


def training(lr, total_dev, slurm_time, transition_schedule):
    return training_config(
        total_train="all", total_dev=total_dev, slurm_time=slurm_time,
        sequence_length=2048, batch_size=8, gradient_accumulation_steps=1,
        gpus="0", report_to="wandb", lr=lr, wd=0.1, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0,
        transition_schedule=transition_schedule)


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

# --- the blocks ---------------------------------------------------------------

# Stage A on the tag side only, at posA's pin.
POS_A = {
    size: {"model": [MODEL_INIT_EXT],
           "transition": [RESET],
           "dataset": [tag_stream(size)],
           "training": [training(PINS_POS[size], total_dev=2_000_000, slurm_time=WALLTIME[size],
                                 transition_schedule=POS_ONLY)],
           "scheduler": [SCHEDULER],
           "evaluation": [eval_config("perplexity_pos_ids", split="dev")]}
    for size in SIZES}

# Stage B with the ramp, at rampB's pin; `transition` is "none" after posA
# (the tag embeddings are the starting point) or a reset from a fresh init.
RAMP_B = {
    (size, transition["TYPE"]): {
        "model": [MODEL_PRETRAINED_EXT],
        "transition": [transition],
        "dataset": [tag_stream(size)],
        "training": [training(PINS_RAMP[size], total_dev=5_000_000, slurm_time=WALLTIME[size],
                              transition_schedule=ramp(size))],
        "scheduler": [SCHEDULER],
        "evaluation": [eval_config("lm_harness", tasks="blimp"),
                       eval_config("perplexity_word_ids_langonly", split="dev", tag_source=TAG_SOURCE)]}
    for size in SIZES for transition in (KEEP_EMBEDDINGS, RESET)}

globals().update(locals())

EXPERIMENTS = {
    "ramp_after_pos_%dM" % (size // 1_000_000): {
        "stage_A": [POS_A[size]],
        "stage_B": [RAMP_B[size, "none"]],
    }
    for size in SIZES}
EXPERIMENTS["ramp_no_prior"] = {
    "stage_A": [NO_STAGE_A],
    "stage_B": [RAMP_B[size, "reset_embeddings"] for size in SIZES],
}
