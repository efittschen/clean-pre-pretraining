"""CONTROL: would any warm start do?  Stage A on unigram-sampled English
words (lexicon, zero syntax), then the same Stage B.  BabyLM line."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config,
                            tokenizer_with_pos_tags_and_dyck)

RANDOM_SEEDS = [42, 43, 44]

# --- tokenizer and corpora ----------------------------------------------------

EXTENDED_TOKENIZER = tokenizer_with_pos_tags_and_dyck(
    "JLTastet/baby-llama-2-345m", tag_source="ptb", num_dyck_symbols=128)

BABYLM = dataset_config("babylm", words=10_000_000, train_words=10_000_000, dev_words=0)
NONE = dataset_config("none", words=2_300_000_000, train_words=2_000_000_000,
                      dev_words=100_000_000, test_words=100_000_000)
# Unigram-sampled real words from the English corpus the language arm reads.
NGRAM = dataset_config(
    "ngram", n=1, source_language="eng_Latn", source_words=4_300_000_000, source_split="trainA",
    source_trainA_words=2_000_000_000, source_trainB_words=2_000_000_000,
    source_dev_words=100_000_000, source_test_words=100_000_000,
    words=12_000_000, train_words=10_000_000, dev_words=1_000_000, test_words=1_000_000)
# Stage B trains on the aligned tag / word layout with P(word) = 1 throughout,
# so its dev stream is the one every arm of the line was scored on.
BABYLM_WORDS = dataset_config("pos_transition", source=BABYLM, tag_source="ptb",
                              words_per_datapoint=115, aligned=True)
WORDS_ONLY = {"shape": "linear", "warmup_words": 0, "transition_words": 0,
              "start_prob": 1.0, "end_prob": 1.0}

# --- model and recipe ---------------------------------------------------------

BABY_MODEL = model_config(architecture_like="JLTastet/baby-llama-2-345m", tokenizer=EXTENDED_TOKENIZER)
RESET = transition_config("reset_embeddings")

STAGE_A_TRAINING = training_config(     # 8M tokens at the line's pinned cell; 1M dev (the ngram dev split)
    checkpoint_interval=[8_000_000], eval_interval=[], total_train=8_000_000, total_dev=1_000_000,
    sequence_length=128, batch_size=128, gradient_accumulation_steps=2, gpus="0", report_to="wandb",
    lr=5e-4, wd=0.1, fp16=True, bf16=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6,
    max_grad_norm=1.0, slurm_time="03:00:00", slurm_memory="16GB")

EPOCH_TOKENS = 14_794_752               # 14.8M rounded down to whole steps of 128 x 128
EVAL_EVERY = 8 * EPOCH_TOKENS // 20     # 20 dev evals across the 8 epochs
STAGE_B_TRAINING = training_config(     # the baby-llama-2 recipe, 8 epochs
    checkpoint_interval=list(range(EPOCH_TOKENS, 8 * EPOCH_TOKENS + 1, EPOCH_TOKENS)),
    eval_interval=list(range(EVAL_EVERY, 20 * EVAL_EVERY + 1, EVAL_EVERY)),
    total_train=14_800_000, total_dev=2_000_000, num_train_epochs=8,
    sequence_length=128, batch_size=128, gradient_accumulation_steps=2, gpus="0", report_to="wandb",
    lr=7e-4, wd=5.0, fp16=True, bf16=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6,
    max_grad_norm=1.0, transition_schedule=WORDS_ONLY, slurm_time="06:00:00", slurm_memory="24GB")

# --- the blocks ---------------------------------------------------------------

NO_STAGE_A = {
    "model": [BABY_MODEL],
    "transition": [RESET],
    "dataset": [NONE],
    "training": [STAGE_A_TRAINING],    # placeholder; never runs
    "scheduler": [scheduler_config("warmup_cosine", warmup_steps=100, min_lr_rate=0.1)],
    "evaluation": [],
}

NGRAM_STAGE_A = {
    "model": [BABY_MODEL],
    "transition": [RESET],
    "dataset": [NGRAM],
    "training": [STAGE_A_TRAINING],
    "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=1.0, min_lr_rate=0.1)],
    "evaluation": [],
}

STAGE_B = {
    "model": [model_config(use_pretrained=True, tokenizer=EXTENDED_TOKENIZER)],
    "transition": [RESET],
    "dataset": [BABYLM_WORDS],
    "training": [STAGE_B_TRAINING],
    "scheduler": [scheduler_config("warmup_cosine", warmup_steps=600, min_lr_rate=0)],
    "evaluation": [eval_config("lm_harness", tasks="blimp"),
                   eval_config("perplexity_word_ids", split="dev")],
}

EXPERIMENTS = {
    "baseline":   {"stage_A": [NO_STAGE_A],    "stage_B": [STAGE_B]},
    "ngram_null": {"stage_A": [NGRAM_STAGE_A], "stage_B": [STAGE_B]},
}
