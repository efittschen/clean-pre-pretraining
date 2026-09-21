"""Build the circuit by hand (synthetic_init.py v2b) instead of grafting it;
Stage B is the usual English."""
from experiment_dsl import (
    model_config, dataset_config, scheduler_config, training_config,
    eval_config, transition_config,
)

RANDOM_SEEDS = [42, 43, 44]

ARCH = "config/300M_infi.json"
BUDGETS = [10, 30, 100]          # Stage B, millions of words

globals().update(locals())

ARMS = [
    # the floor: a fresh host with no circuit, built the same way
    model_config(graft=dict(donor=None, arch=ARCH, name="synthetic_baseline")),
    model_config(synthetic=dict(version="2b", arch=ARCH, name="synthetic")),
]


def _stage_b_dataset(words_M):
    return dataset_config(
        "language", language="eng_Latn", words=4_300_000_000,
        trainA_words=2_000_000_000, trainB_words=2_000_000_000,
        dev_words=100_000_000, test_words=100_000_000,
        train_split="trainB", tokenizer_split="trainB",
        train_prefix=words_M * 1_000_000,
    )


def _training():
    return training_config(
        total_train="all", total_dev=2_000_000,
        sequence_length=128, batch_size=128,
        gradient_accumulation_steps=2, gpus="0", report_to="none",
        lr=7e-4, wd=5.0, bf16=True, fp16=False,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6, max_grad_norm=1.0,
    )


SCHEDULER = [scheduler_config("warmup_cosine", warmup_steps=600, min_lr_rate=0.0)]

globals().update(locals())

EXPERIMENTS = {}
for _T in BUDGETS:
    EXPERIMENTS["synthetic_B%dM" % _T] = {
        "stage_A": [{
            "model": ARMS,
            "dataset": [dataset_config("none", words=12_000_000,
                                       train_words=10_000_000,
                                       dev_words=1_000_000, test_words=1_000_000)],
            "training": [_training()],
            "scheduler": SCHEDULER,
            "transition": [transition_config("none")],
            "evaluation": [],
        }],
        "stage_B": [{
            "model": [model_config(use_pretrained=True, tokenizer="train",
                                   vocab_size=16_000)],
            "transition": [transition_config("reset_embeddings")],
            "dataset": [_stage_b_dataset(_T)],
            "training": [_training()],
            "scheduler": SCHEDULER,
            "evaluation": [eval_config("perplexity", split="dev")],
        }],
    }
