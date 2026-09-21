"""The one trainer.  Modes are selected by config keys: TRAIN_ALL, MODEL_LOAD,
RESCALE_STD, LOG_EVAL, CKPT_PROBE, FREEZE_LAYERS, EMA_TAUS_STEPS, TRANSITION_SCHEDULE."""
import inspect
import json
import os
import re
import subprocess
import sys

import fire
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    IntervalStrategy,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from dataloader import GBDataset  # noqa: E402

import matrix_scales  # noqa: E402
import probe as probe_mod  # noqa: E402
import rescale as rescale_mod  # noqa: E402

# Match HF checkpoint dirs exactly ("checkpoint-10147") and reject pipeline
# transition dirs ("checkpoint-10147-transition").
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _resolve_load_kwargs(model_load):
    kwargs = dict(model_load or {})
    dtype = kwargs.pop("torch_dtype", None)
    if dtype is not None:
        if dtype not in _DTYPES:
            raise ValueError(f"MODEL_LOAD.torch_dtype={dtype!r} unknown; use one of {sorted(_DTYPES)}")
        kwargs["torch_dtype"] = _DTYPES[dtype]
    return kwargs


def _freeze_body(model, cfg):
    num_layers = model.config.num_hidden_layers
    bottom = cfg.get("UNFROZEN_BOTTOM", 0)
    top = cfg.get("UNFROZEN_TOP", 0)
    frozen_range = range(bottom, num_layers - top)
    for name, param in model.named_parameters():
        for i in frozen_range:
            if f".layers.{i}." in name:
                param.requires_grad = False
                break
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Froze layers {bottom}-{num_layers - top - 1} "
          f"({len(frozen_range)}/{num_layers}). "
          f"Trainable: {trainable}/{total} ({100 * trainable / total:.1f}%)")


def _freeze_final_norm(model):
    names = [n for n, _ in model.named_parameters() if n == "model.norm.weight"]
    assert names, "FREEZE_FINAL_NORM is set but this architecture has no model.norm.weight"
    for name, param in model.named_parameters():
        if name in names:
            param.requires_grad = False
    print(f"Froze final norm ({', '.join(names)})")
    print("Trainable parameters: " + ", ".join(
        f"{n} ({p.numel()})" for n, p in model.named_parameters() if p.requires_grad))


def _log_gpu_state():
    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=10, check=False)
    except FileNotFoundError:
        print("nvidia-smi not available; skipping the GPU report")
        return
    print("nvidia-smi output:\n", r.stdout)
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       text=True, capture_output=True, check=False)
    for pid in [x.strip() for x in p.stdout.splitlines() if x.strip().isdigit()]:
        u = subprocess.run(["ps", "-o", "user=", "-p", pid], text=True,
                           capture_output=True, check=False)
        print(f"PID {pid} owned by user {u.stdout.strip() or 'unknown'}")


def _latest_resumable_checkpoint(output_dir):
    found = []
    for d in os.listdir(output_dir):
        m = _CHECKPOINT_RE.match(d)
        if not m:
            continue
        p = os.path.join(output_dir, d)
        if not os.path.isdir(p):
            continue
        has_state = os.path.isfile(os.path.join(p, "trainer_state.json"))
        has_weights = (os.path.isfile(os.path.join(p, "model.safetensors"))
                       or os.path.isfile(os.path.join(p, "pytorch_model.bin")))
        if has_state and has_weights:
            found.append((int(m.group(1)), d))
    if not found:
        return None
    found.sort()
    return os.path.join(output_dir, found[-1][1])


def _init_wandb(training_cfg, scheduler_cfg, training_args, inputs, **extra):
    import wandb
    kwargs = dict(
        project=training_cfg["HUGGINGFACE_CONFIG"].get("log_project", "pre_pretraining"),
        name=training_cfg["HUGGINGFACE_CONFIG"].get("log_name", inputs["output_dir"].split("/")[-1]),
        config={"training_cfg": training_cfg,
                "scheduler_cfg": scheduler_cfg, "training_args": training_args.to_dict(),
                "function_inputs": inputs, **extra},
    )
    try:
        wandb.init(**kwargs)
    except wandb.errors.CommError:
        print("wandb online init failed, falling back to offline mode")
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(**kwargs)


def _write_plateau_summary(trainer, final_dir, cfg, huggingface_config,
                           filtered_config, train_windows):
    state = trainer.state
    batch_tokens = cfg["SEQUENCE_LENGTH"] * cfg["BATCH_SIZE"]
    best_step = None
    if state.best_model_checkpoint:
        m = _CHECKPOINT_RE.match(os.path.basename(state.best_model_checkpoint))
        best_step = int(m.group(1)) if m else None
    epoch_cap = float(huggingface_config.get("num_train_epochs", 1))
    summary = {
        "best_eval_loss": state.best_metric,
        "best_step": best_step,
        "best_tokens": best_step * batch_tokens if best_step is not None else None,
        "best_epoch": (round(best_step / max(1, state.global_step) * state.epoch, 3)
                       if best_step and state.epoch else None),
        "stop_step": state.global_step,
        "stop_epoch": state.epoch,
        "stop_tokens": state.global_step * batch_tokens,
        "stopped_early": bool(state.epoch is not None and state.epoch < epoch_cap - 1e-6),
        "epoch_cap": epoch_cap,
        "early_stopping_patience": int(huggingface_config.get("early_stopping_patience", 5)),
        "eval_steps": filtered_config.get("eval_steps"),
        "warmup_steps": int(huggingface_config.get("warmup_steps", 0) or 0),
        "lr_scheduler_kwargs": filtered_config.get("lr_scheduler_kwargs"),
        "train_windows": train_windows,
        "trace": [{"step": h["step"], "eval_loss": h.get("eval_loss"),
                   "lr": h.get("learning_rate"), "epoch": h.get("epoch")}
                  for h in state.log_history if "eval_loss" in h or "learning_rate" in h],
    }
    with open(os.path.join(final_dir, "plateau_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved BEST model (dev loss {state.best_metric}) to {final_dir}; "
          f"stopped at step {state.global_step} / epoch {state.epoch}")


def _tok_datasets(train_data, dev_data, cfg, tokenizer):
    seq_length = cfg["SEQUENCE_LENGTH"]
    bos_id = tokenizer.bos_token_id if cfg.get("USE_BOS", False) else None
    train_dataset = GBDataset(train_data, seq_length,
                              random_chunk=cfg.get("RANDOM_CHUNK", False),
                              bos_token_id=bos_id)
    eval_dataset = GBDataset(dev_data, seq_length, random_chunk=False, bos_token_id=bos_id)

    if cfg.get("TRAIN_ALL"):
        # Token count is the per-run outcome of the word budget: consume every window.
        assert len(train_dataset) > 0, f"Empty training dataset: {train_data}"
        print(f"Train dataset (ALL windows): {len(train_dataset)} x {seq_length} "
              f"= {len(train_dataset) * seq_length} tokens")
    else:
        train_sequences = cfg["TRAIN_TOKENS"] // seq_length
        assert train_sequences > 0 and len(train_dataset) >= train_sequences, \
            f"Not enough training data sequences {len(train_dataset)} < {train_sequences}"
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_sequences)))

    dev_sequences = cfg["DEV_TOKENS"] // seq_length
    assert dev_sequences > 0 and len(eval_dataset) >= dev_sequences, \
        f"Not enough dev data sequences {len(eval_dataset)} < {dev_sequences}"
    eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(dev_sequences)))
    return train_dataset, eval_dataset


def train_model(model, output_dir, train_data, dev_data, scheduler_config,
                training_config, dataset_config, tokenizer_path,
                flag, random_seed=42):
    with open(training_config) as f:
        training_cfg = json.load(f)
    with open(scheduler_config) as f:
        scheduler_cfg = json.load(f)
    with open(dataset_config) as f:
        dataset_cfg = json.load(f)

    cfg = training_cfg["CONFIG"]
    seq_length = cfg["SEQUENCE_LENGTH"]
    train_all = bool(cfg.get("TRAIN_ALL"))
    rescale_std = cfg.get("RESCALE_STD")
    use_probe = bool(cfg.get("CKPT_PROBE"))
    log_eval = bool(cfg.get("LOG_EVAL"))
    plateau = train_all and scheduler_cfg.get("TYPE") == "reduce_lr_on_plateau"
    pos_transition = dataset_cfg.get("TYPE") == "pos_transition"

    if scheduler_cfg.get("TYPE", "").startswith("custom_"):
        raise ValueError(
            "custom_* schedulers were removed 2026-09-21 with the optimizer ablation "
            "(scripts/model/schedulers.py); use warmup_cosine, wsd or reduce_lr_on_plateau.")
    if pos_transition and (use_probe or cfg.get("EMA_TAUS_STEPS")):
        # Both read a flat int64 .tok dev stream, not the aligned CSR streams.
        raise NotImplementedError(
            "CKPT_PROBE and EMA_TAUS_STEPS are only implemented for .tok streams, "
            "not for the pos_transition (aligned CSR) data.")

    # The optimizer is the Trainer's AdamW; its knobs are ordinary training keys.
    huggingface_config = {**training_cfg["HUGGINGFACE_CONFIG"],
                          **scheduler_cfg.get("HUGGINGFACE_CONFIG", {})}
    huggingface_config["output_dir"] = output_dir
    huggingface_config["seed"] = random_seed

    model_path = model
    model = AutoModelForCausalLM.from_pretrained(model, **_resolve_load_kwargs(cfg.get("MODEL_LOAD", {})))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Fixed forward multipliers, if this model carries a matrix_scales.json.
    # Self-gating: a model without one is untouched.
    matrix_scales.install(model, model_path)

    ema_cb = None
    if cfg.get("EMA_TAUS_STEPS"):
        import ema as ema_mod
        ema_cb = ema_mod.EMACallback(
            model, cfg["EMA_TAUS_STEPS"], dev_data, cfg["DEV_TOKENS"],
            seq_length, cfg.get("EVAL_EVERY_STEPS", 1852),
            cfg.get("PATIENCE_EVALS", 2), output_dir, model_path,
            track_tokens=cfg.get("EMA_TRACK_TOKENS"))

    # --- optional body reparametrization (W = c * W') --------------------
    rescale_constants = None
    if rescale_std is not None:
        rescale_constants = rescale_mod.apply_rescale(model, rescale_std)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rescale_constants.json"), "w") as f:
            json.dump({"rescale_std": rescale_std, "constants": rescale_constants}, f, indent=2)
        cs = sorted(rescale_constants.values())
        print(f"[rescale] {len(cs)} body matrices -> std {rescale_std}; "
              f"c min {cs[0]:.3f} median {cs[len(cs) // 2]:.3f} max {cs[-1]:.3f}", flush=True)

    if cfg.get("FREEZE_LAYERS", False):
        _freeze_body(model, cfg)
    if cfg.get("FREEZE_FINAL_NORM", False):
        _freeze_final_norm(model)

    # --- data -------------------------------------------------------------
    callbacks = []
    wandb_extra = {}
    if pos_transition:
        import pos_transition as pt
        train_dataset, eval_dataset, pt_info = pt.build_datasets(
            train_data, dev_data, cfg, dataset_cfg, random_seed)
        callbacks.append(pt.StepUpdateCallback(train_dataset))
        wandb_extra = {"transition_schedule": pt_info["transition_schedule"],
                       "words_per_datapoint": pt_info["words_per_datapoint"]}
    else:
        train_dataset, eval_dataset = _tok_datasets(train_data, dev_data, cfg, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- training arguments ----------------------------------------------
    filtered_config = {k: v for k, v in huggingface_config.items()
                       if k in inspect.signature(TrainingArguments.__init__).parameters}
    print(f"Not using training args: {[k for k in huggingface_config if k not in filtered_config]}")

    if plateau:
        # "final" means BEST-dev for plateau runs; the downstream graph still
        # keys on checkpoint-final, so the contract is unchanged.
        filtered_config["load_best_model_at_end"] = True
        filtered_config["metric_for_best_model"] = "eval_loss"
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(huggingface_config.get("early_stopping_patience", 5))))

    if isinstance(eval_dataset, dict) and filtered_config.get("load_best_model_at_end"):
        # With a dict of eval sets the best-model metric must name one; the word side is the target.
        if filtered_config.get("metric_for_best_model", "eval_loss") == "eval_loss":
            filtered_config["metric_for_best_model"] = "eval_word_loss"

    probe_state = None
    if use_probe:
        bin_of, bin_meta = probe_mod.build_freq_bins(train_data, len(tokenizer))
        probe_dir = os.path.join(output_dir, "probe")
        probe_state = (bin_of, bin_meta, probe_dir)

    if filtered_config.get("save_strategy") == "custom":
        checkpoint_steps = cfg["CHECKPOINT_LIST_STEPS"]
        evaluation_steps = cfg.get("EVALUATION_LIST_STEPS", [])

        if use_probe:
            class ProbeCheckpointCallback(TrainerCallback):

                def __init__(self, steps):
                    self.steps = set(steps)
                    self.done = set()

                def on_step_end(self, args, state, control, model=None, **kwargs):
                    if state.global_step in self.steps and state.global_step not in self.done:
                        self.done.add(state.global_step)
                        control.should_save = True
                        probe_mod.run_probe(model, state.global_step, eval_dataset,
                                            seq_length, *probe_state)
                    return control

            callbacks.append(ProbeCheckpointCallback(evaluation_steps or checkpoint_steps))
        else:
            class IrregularCheckpointCallback(TrainerCallback):
                def __init__(self, checkpoint_steps, evaluation_steps=()):
                    self.checkpoint_steps = set(checkpoint_steps)
                    self.evaluation_steps = set(evaluation_steps).union(self.checkpoint_steps)

                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step in self.evaluation_steps:
                        control.should_log = True
                        control.should_evaluate = True
                        if state.global_step in self.checkpoint_steps:
                            control.should_save = True
                    return control

            callbacks.append(IrregularCheckpointCallback(checkpoint_steps, evaluation_steps))

        filtered_config["eval_strategy"] = IntervalStrategy.NO
        filtered_config["save_strategy"] = IntervalStrategy.NO

    if ema_cb is not None:
        callbacks.append(ema_cb)

    print(f"filtered config: {filtered_config}")
    training_args = TrainingArguments(**filtered_config)
    print(f"[seed check] CLI random_seed={random_seed} -> "
          f"TrainingArguments.seed={training_args.seed}", flush=True)

    _log_gpu_state()

    if training_cfg["HUGGINGFACE_CONFIG"].get("report_to") == "wandb":
        _init_wandb(training_cfg, scheduler_cfg, training_args,
                    {"model": model_path, "output_dir": output_dir,
                     "train_data": train_data, "dev_data": dev_data,
                     "random_seed": random_seed, "tokenizer": tokenizer_path,
                     "scheduler_config": scheduler_config,
                     "training_config": training_config},
                    **wandb_extra)

    # --- trainer ----------------------------------------------------------
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      callbacks=callbacks)

    resume = _latest_resumable_checkpoint(output_dir)
    if resume:
        print(f"Resuming training from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # --- save -------------------------------------------------------------
    if rescale_std is not None:
        # Bake W = c * W' back in so the checkpoint is a plain HF model.
        rescale_mod.bake_rescale(trainer.model)

    trainer.save_model(filtered_config["output_dir"])
    if train_all:
        # Step count is data-dependent in train-all mode, so downstream eval
        # targets key on this fixed name.
        final_dir = os.path.join(filtered_config["output_dir"], "checkpoint-final")
        trainer.save_model(final_dir)
        print(f"Saved final model to {final_dir} after {trainer.state.global_step} steps")
        if plateau:
            _write_plateau_summary(trainer, final_dir, cfg, huggingface_config,
                                   filtered_config, len(train_dataset))

    matrix_scales.copy_into(model_path, os.path.join(filtered_config["output_dir"],
                                                     "checkpoint-final"))
    if ema_cb is not None:
        ema_cb.save_all(trainer.model)

    if use_probe:
        probe_mod.run_probe(trainer.model, trainer.state.global_step, eval_dataset,
                            seq_length, *probe_state)

    if log_eval:
        path = os.path.join(filtered_config["output_dir"], "log_history.json")
        with open(path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Wrote {len(trainer.state.log_history)} log_history entries to {path}")

    with open(flag, "w") as f:
        f.write("Training finished successfully.\n")


if __name__ == "__main__":
    fire.Fire(train_model)
