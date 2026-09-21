"""Hu et al. 2025 2x2 reproduction, our-code half: shuffle-Dyck / C4 / none
into C4 on Pythia-160m, under the hyperparameters in Hu's code and in the
paper's appendix."""
from experiment_dsl import (model_config, dataset_config, scheduler_config,
                            training_config, eval_config, transition_config)
from site_config import grad_accum_for

RANDOM_SEEDS = [42, 43, 44]
STAGE_A_TOKENS = 32_000_000
STAGE_B_TOKENS = 665_000_000

# "code": what from_hu's code runs; "paper": what its appendix says.
VARIANTS = {
    "actually_original_160M_code":  dict(lr=1e-3, wd=0.0, adam_epsilon=1e-8, warmup_steps=500),
    "actually_original_160M_paper": dict(lr=5e-4, wd=0.1, adam_epsilon=1e-6, warmup_steps=1000),
}

# --- corpora ------------------------------------------------------------------

DYCK = dataset_config("shuffle_dyck", num_symbols=128, target_length=2048, words=2_300_000_000,
                      train_words=2_000_000_000, dev_words=100_000_000, test_words=100_000_000)
C4 = dataset_config("c4", c4_config="en", words=2_300_000_000,
                    train_words=2_000_000_000, dev_words=100_000_000, test_words=100_000_000)
NONE = dataset_config("none", words=2_300_000_000,
                      train_words=2_000_000_000, dev_words=100_000_000, test_words=100_000_000)

# --- model and recipe ---------------------------------------------------------

PYTHIA_INIT = model_config(architecture_like="EleutherAI/pythia-160m", tokenizer="EleutherAI/pythia-160m")
PYTHIA_PRETRAINED = model_config(use_pretrained=True, tokenizer="EleutherAI/pythia-160m")
RESET = transition_config("reset_embeddings")


def training(tokens, lr, wd, adam_epsilon):
    return training_config(
        checkpoint_interval=[tokens], eval_interval=[], total_train=tokens, total_dev=20_000_000,
        sequence_length=2048, batch_size=32, gradient_accumulation_steps=grad_accum_for(32, "160M"),
        gpus="0", report_to="wandb", lr=lr, wd=wd, fp16=False, bf16=True,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=adam_epsilon, max_grad_norm=1.0,
        model_load={"torch_dtype": "bfloat16"})      # bf16 master weights, as from_hu


globals().update(locals())

# --- the blocks: one experiment per variant -----------------------------------

EXPERIMENTS = {
    name: {
        "stage_A": [{"model": [PYTHIA_INIT],
                     "transition": [RESET],
                     "dataset": [DYCK, C4, NONE],
                     "training": [training(STAGE_A_TOKENS, v["lr"], v["wd"], v["adam_epsilon"])],
                     "scheduler": [scheduler_config("warmup_cosine", warmup_steps=v["warmup_steps"],
                                                    min_lr_rate=0.1)],
                     "evaluation": []}],
        "stage_B": [{"model": [PYTHIA_PRETRAINED],
                     "transition": [RESET],
                     "dataset": [C4],
                     "training": [training(STAGE_B_TOKENS, v["lr"], v["wd"], v["adam_epsilon"])],
                     "scheduler": [scheduler_config("warmup_cosine", warmup_steps=v["warmup_steps"],
                                                    min_lr_rate=0.1)],
                     "evaluation": [eval_config("lm_harness", tasks="blimp"),
                                    eval_config("perplexity", split="test")]}],
    }
    for name, v in VARIANTS.items()}
