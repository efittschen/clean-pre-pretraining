"""DEMO 3 of 3: the POS ramp at 1/100 scale.  One seed, about an hour on one GPU.

    scons EXPERIMENT=configs/demo/02_pos_ramp.py -n
    scons EXPERIMENT=configs/demo/02_pos_ramp.py

Every word of a 1M-word English prefix is tagged, so Stage B can emit each
word either as its subword tokens or as its PTB tag token.  P(word) ramps
from 0 to 1 over the first 10 percent of training and stays at 1.  Three
arms on the same words, the same trained tokenizer and the same model:

    ramp_after_pos   Stage A on the tag side only (P(word) = 0), embeddings
                     kept, then the ramp.  The treatment.
    ramp_no_prior    the ramp from a fresh init.
    plain            ordinary training on the same words.  The baseline.

"""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config,
                            tokenizer_with_pos_tags)

RANDOM_SEEDS = [42]

WORDS = 1_000_000               # the Stage-B budget, in words of the corpus
RAMP_WORDS = 100_000            # the ramp window: the first 10 percent
LR = 1e-3                       # every arm; the real unit pins each from its search

# --- corpus: 3M words of FineWeb English, of which the first 1M of train are used

def english(**aligned):
    return dataset_config(
        "language" if not aligned else "pos_transition", language="eng_Latn",
        words=3_000_000, train_words=2_500_000, dev_words=250_000, test_words=250_000,
        train_split="train", tokenizer_split="train", train_prefix=WORDS, **aligned)


globals().update(locals())

PLAIN_STREAM = english()
# The same words with their PTB tags aligned to the subword tokens, four CSR
# files per split; dev_prefix caps the tagging of the dev split.
TAG_STREAM = english(tag_source="ptb", aligned=True, dev_prefix=250_000,
                     words_per_datapoint=400)   # stride in words (~1.3 tokens/word x 512)

POS_ONLY = {"shape": "linear", "warmup_words": 0, "transition_words": 0,
            "start_prob": 0.0, "end_prob": 0.0}
RAMP = {"shape": "linear", "warmup_words": 0, "transition_words": RAMP_WORDS,
        "start_prob": 0.0, "end_prob": 1.0}

# --- models: a Pythia-160m body; the ramp arms' tokenizer is the plain one
#     trained on the prefix, extended with 87 marker-prefixed PTB tags ------

EXT_TOKENIZER = tokenizer_with_pos_tags("train", tag_source="ptb", marker=True)
MODEL_INIT = model_config(architecture_like="EleutherAI/pythia-160m", tokenizer="train", vocab_size=16_000)
MODEL_INIT_EXT = model_config(architecture_like="EleutherAI/pythia-160m", tokenizer=EXT_TOKENIZER, vocab_size=16_000)
MODEL_PRETRAINED = model_config(use_pretrained=True, tokenizer="train", vocab_size=16_000)
MODEL_PRETRAINED_EXT = model_config(use_pretrained=True, tokenizer=EXT_TOKENIZER, vocab_size=16_000)
RESET = transition_config("reset_embeddings")
KEEP_EMBEDDINGS = transition_config("none")
SCHEDULER = scheduler_config("warmup_cosine", warmup_ratio=0.1, min_lr_rate=0.1)


def training(total_dev, transition_schedule=None):
    return training_config(
        total_train="all", total_dev=total_dev, slurm_time="01:00:00",
        sequence_length=512, batch_size=8, gradient_accumulation_steps=1,
        gpus="0", report_to="none", lr=LR, wd=0.1, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0,
        transition_schedule=transition_schedule)


# --- the blocks ---------------------------------------------------------------

NO_STAGE_A = {
    "model": [MODEL_INIT],
    "transition": [RESET],
    "dataset": [dataset_config("none")],
    "training": [training(total_dev=200_000)],     # placeholder; never runs
    "scheduler": [SCHEDULER],
    "evaluation": [],
}

POS_A = {
    "model": [MODEL_INIT_EXT],
    "transition": [RESET],
    "dataset": [TAG_STREAM],
    "training": [training(total_dev=100_000, transition_schedule=POS_ONLY)],
    "scheduler": [SCHEDULER],
    "evaluation": [eval_config("perplexity_pos_ids", split="dev")],
}

RAMP_B_AFTER_POS = {
    "model": [MODEL_PRETRAINED_EXT],
    "transition": [KEEP_EMBEDDINGS],
    "dataset": [TAG_STREAM],
    "training": [training(total_dev=200_000, transition_schedule=RAMP)],
    "scheduler": [SCHEDULER],
    "evaluation": [eval_config("perplexity_word_ids_langonly", split="dev", tag_source="ptb")],
}

RAMP_B_FRESH = {**RAMP_B_AFTER_POS, "transition": [RESET]}

PLAIN_B = {
    "model": [MODEL_PRETRAINED],
    "transition": [RESET],
    "dataset": [PLAIN_STREAM],
    "training": [training(total_dev=200_000)],
    "scheduler": [SCHEDULER],
    "evaluation": [eval_config("perplexity", split="dev")],
}

EXPERIMENTS = {
    "ramp_after_pos": {"stage_A": [POS_A],      "stage_B": [RAMP_B_AFTER_POS]},
    "ramp_no_prior":  {"stage_A": [NO_STAGE_A], "stage_B": [RAMP_B_FRESH]},
    "plain":          {"stage_A": [NO_STAGE_A], "stage_B": [PLAIN_B]},
}
