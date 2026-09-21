"""DEMO 2 of 2: a shuffled-Dyck prior vs no prior into 10M words of
English, one seed, an afternoon."""
from experiment_dsl import (
    model_config, dataset_config, scheduler_config, training_config,
    eval_config, transition_config,
)

RANDOM_SEEDS = [42]

_A_TOKENS = 10_000_000
_B_TOKENS = 10_000_000

_TRAINING = dict(
    total_dev=1_000_000,
    sequence_length=512,
    batch_size=16,
    gradient_accumulation_steps=1,
    gpus="0",
    report_to="none",
    wd=0.1,
    bf16=True,
    fp16=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
)

_ENGLISH = dict(language="eng_Latn", words=30_000_000,
                train_words=25_000_000, dev_words=2_500_000, test_words=2_500_000)

EXPERIMENTS = {
    "prior_vs_baseline": {
        "stage_A": [
            {
                "model": [model_config(architecture_like="EleutherAI/pythia-160m",
                                       tokenizer="EleutherAI/pythia-160m")],
                "transition": [transition_config("reset_embeddings")],
                "dataset": [
                    # The baseline arm: no Stage A at all.
                    dataset_config("none", words=12_000_000, train_words=10_000_000,
                                   dev_words=1_000_000, test_words=1_000_000),
                    # The treatment arm: bracket structure, no words.
                    dataset_config("shuffle_dyck", num_symbols=128, target_length=512,
                                   words=12_000_000, train_words=10_000_000,
                                   dev_words=1_000_000, test_words=1_000_000),
                ],
                "training": [training_config(checkpoint_interval=[_A_TOKENS],
                                             eval_interval=[], total_train=_A_TOKENS,
                                             lr=1e-3, **_TRAINING)],
                "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=0.1,
                                               min_lr_rate=0.1)],
                "evaluation": [],
            }
        ],
        "stage_B": [
            {
                "model": [model_config(use_pretrained=True,
                                       tokenizer="EleutherAI/pythia-160m")],
                "transition": [transition_config("reset_embeddings")],
                "dataset": [dataset_config("language", **_ENGLISH)],
                "training": [training_config(checkpoint_interval=[_B_TOKENS],
                                             eval_interval=[], total_train=_B_TOKENS,
                                             lr=5e-4, **_TRAINING)],
                "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=0.1,
                                               min_lr_rate=0.1)],
                "evaluation": [eval_config("perplexity", split="dev")],
            }
        ],
    },
}
