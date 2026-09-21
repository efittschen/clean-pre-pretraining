"""DEMO 1 of 2: does the graph run end to end?  Minutes on one GPU,
no corpus download, no science."""
from experiment_dsl import (
    model_config, dataset_config, scheduler_config, training_config,
    eval_config, transition_config,
)

RANDOM_SEEDS = [42]

# Tiny budgets: ~2M tokens a stage at seq 512 / batch 8 is a few hundred steps.
_A_TOKENS = 100_000
_B_TOKENS = 100_000

_TRAINING = dict(
    total_dev=200_000,
    sequence_length=512,
    batch_size=8,
    gradient_accumulation_steps=1,   # hardcoded: grad_accum_for() needs ONE partition
    gpus="0",
    report_to="none",
    lr=5e-4,
    wd=0.1,
    bf16=False,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
)

_DYCK = dict(num_symbols=32, target_length=512, words=6_000_000,
             train_words=5_000_000, dev_words=500_000, test_words=500_000)

EXPERIMENTS = {
    "smoke": {
        "stage_A": [
            {
                "model": [model_config(architecture_like="EleutherAI/pythia-160m",
                                       tokenizer="EleutherAI/pythia-160m")],
                "transition": [transition_config("reset_embeddings")],
                "dataset": [dataset_config("shuffle_dyck", **_DYCK)],
                "training": [training_config(checkpoint_interval=[_A_TOKENS],
                                             eval_interval=[],
                                             total_train=_A_TOKENS, **_TRAINING)],
                "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=0.1,
                                               min_lr_rate=0.1)],
                "evaluation": [],
            }
        ],
        "stage_B": [
            {
                # use_pretrained=True -> carry Stage A's body through the transition
                "model": [model_config(use_pretrained=True,
                                       tokenizer="EleutherAI/pythia-160m")],
                "transition": [transition_config("reset_embeddings")],
                "dataset": [dataset_config("shuffle_dyck", **_DYCK)],
                "training": [training_config(checkpoint_interval=[_B_TOKENS],
                                             eval_interval=[],
                                             total_train=_B_TOKENS, **_TRAINING)],
                "scheduler": [scheduler_config("warmup_cosine", warmup_ratio=0.1,
                                               min_lr_rate=0.1)],
                "evaluation": [eval_config("perplexity", split="test")],
            }
        ],
    },
}
