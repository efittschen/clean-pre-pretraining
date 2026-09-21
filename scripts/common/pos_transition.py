"""The POS -> word ramp as a dataset for train.py: four CSR memmaps per
split, per-word coin flips under TRANSITION_SCHEDULE."""
import math
import multiprocessing

import numpy as np
import torch
from transformers import TrainerCallback

STREAM_KEYS = ("word_ids", "word_offsets", "pos_ids", "pos_offsets")
_WORD_IDS_SUFFIX = ".word_ids.bin"


def stream_files(word_ids_path):
    if not word_ids_path.endswith(_WORD_IDS_SUFFIX):
        raise ValueError(f"pos_transition data must be the .word_ids.bin path, got {word_ids_path!r}")
    prefix = word_ids_path[:-len(_WORD_IDS_SUFFIX)]
    return {k: f"{prefix}.{k}.bin" for k in STREAM_KEYS}


def make_schedule_fn(schedule_cfg, total_steps, words_per_step=None):
    shape = schedule_cfg.get("shape", "linear")
    start_prob = float(schedule_cfg.get("start_prob", 0.0))
    end_prob = float(schedule_cfg.get("end_prob", 1.0))

    warmup_words = schedule_cfg.get("warmup_words")
    transition_words = schedule_cfg.get("transition_words")
    word_based = (warmup_words is not None) or (transition_words is not None)

    if word_based:
        if words_per_step is None or words_per_step <= 0:
            raise ValueError(
                "transition_schedule uses word-based keys "
                "(warmup_words / transition_words) but words_per_step "
                f"was not provided (got {words_per_step}).")
        warmup_words = int(warmup_words or 0)
        if transition_words is None:
            raise ValueError(
                "transition_schedule with warmup_words must also set "
                "transition_words (the ramp window length in source words).")
        transition_words = int(transition_words)
        start_step = warmup_words // int(words_per_step)
        end_step = start_step + max(0, transition_words // int(words_per_step))
    else:
        start_step = int(schedule_cfg.get("start_step", 0))
        end_step = schedule_cfg.get("end_step", "auto")
        if end_step == "auto" or end_step is None:
            end_step = int(total_steps)
        end_step = int(end_step)

    def fn(step):
        if step <= start_step or end_step <= start_step:
            return start_prob
        if step >= end_step:
            return end_prob
        t = (step - start_step) / (end_step - start_step)
        if shape == "cosine":
            t = (1.0 - math.cos(t * math.pi)) / 2.0
        return start_prob + (end_prob - start_prob) * t

    if word_based:
        print(f"[pos_transition] schedule: shape={shape}  "
              f"warmup_words={warmup_words:,}  "
              f"transition_words={transition_words:,}  "
              f"-> steps[{start_step}->{end_step}]  "
              f"prob[{start_prob}->{end_prob}]  "
              f"(words_per_step={words_per_step:,}, "
              f"total_steps={total_steps})")
    else:
        print(f"[pos_transition] schedule: shape={shape}  "
              f"steps[{start_step}->{end_step}]  "
              f"prob[{start_prob}->{end_prob}]  "
              f"(total_steps={total_steps})")
    return fn


class PosTransitionDataset(torch.utils.data.Dataset):

    def __init__(self, files, seq_length, words_per_datapoint,
                 schedule_fn=None, fixed_prob_word=None,
                 base_seed=42, num_examples=None):
        self.word_ids = np.memmap(files["word_ids"], dtype=np.int32, mode="r")
        self.word_off = np.memmap(files["word_offsets"], dtype=np.int64, mode="r")
        self.pos_ids = np.memmap(files["pos_ids"], dtype=np.int32, mode="r")
        self.pos_off = np.memmap(files["pos_offsets"], dtype=np.int64, mode="r")
        self.seq_length = int(seq_length)
        self.N = int(words_per_datapoint)
        self.schedule_fn = schedule_fn
        self.fixed_prob_word = fixed_prob_word
        self.base_seed = int(base_seed)

        # Written by the main-process callback, read anywhere.
        self._step_shared = multiprocessing.Value("q", 0)

        # An example consumes at most seq_length source words.
        total_source_words = len(self.word_off) - 1
        max_examples = max(0, (total_source_words - self.seq_length) // self.N)
        if num_examples is not None:
            max_examples = min(max_examples, int(num_examples))
        self._len = max_examples

        assert (self.schedule_fn is None) != (self.fixed_prob_word is None), (
            "Provide exactly one of schedule_fn (training) "
            "or fixed_prob_word (eval).")

    @staticmethod
    def source_words(files):
        return len(np.memmap(files["word_offsets"], dtype=np.int64, mode="r")) - 1

    def __len__(self):
        return self._len

    def set_step(self, step):
        with self._step_shared.get_lock():
            self._step_shared.value = int(step)

    def _current_step(self):
        return int(self._step_shared.value)

    def _prob_word(self, step):
        if self.fixed_prob_word is not None:
            return float(self.fixed_prob_word)
        return float(self.schedule_fn(step))

    def __getitem__(self, i):
        step = self._current_step()
        prob_word = self._prob_word(step)
        src = i * self.N

        # (seed, step, i) determines the mask.
        h = ((self.base_seed & 0xFFFFFFFF) * 2654435761
             ^ ((step + 1) & 0xFFFFFFFF) * 40503
             ^ ((i + 1) & 0xFFFFFFFF) * 2246822519)
        rng = np.random.default_rng(h & 0xFFFFFFFFFFFFFFFF)

        spans = []
        total = 0
        j = src
        chunk = max(64, self.seq_length // 2)
        randoms = rng.random(chunk)
        ridx = 0

        while total < self.seq_length:
            if ridx >= chunk:
                randoms = rng.random(chunk)
                ridx = 0
            use_word = randoms[ridx] < prob_word
            ridx += 1

            if use_word:
                start = int(self.word_off[j])
                end = int(self.word_off[j + 1])
                span = self.word_ids[start:end]
            else:
                start = int(self.pos_off[j])
                end = int(self.pos_off[j + 1])
                span = self.pos_ids[start:end]
            spans.append(span)
            total += span.shape[0]
            j += 1

        ids = np.concatenate(spans)[:self.seq_length]
        return torch.from_numpy(ids.astype(np.int64))


class StepUpdateCallback(TrainerCallback):

    def __init__(self, dataset_ref):
        self.dataset_ref = dataset_ref

    def _current_prob_word(self, step):
        if self.dataset_ref.fixed_prob_word is not None:
            return float(self.dataset_ref.fixed_prob_word)
        return float(self.dataset_ref.schedule_fn(step))

    def _log_prob_word(self, step):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.log({"transition/prob_word": self._current_prob_word(step)}, step=step)

    def on_train_begin(self, args, state, control, **kwargs):
        self.dataset_ref.set_step(state.global_step)
        self._log_prob_word(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.dataset_ref.set_step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # User callbacks run after WandbCallback, so push directly too.
        prob = self._current_prob_word(state.global_step)
        if logs is not None:
            logs["transition/prob_word"] = prob
        self._log_prob_word(state.global_step)


def build_datasets(train_word_ids, dev_word_ids, cfg, dataset_cfg, random_seed):
    seq_length = cfg["SEQUENCE_LENGTH"]
    batch_size = cfg["BATCH_SIZE"]
    train_files = stream_files(train_word_ids)
    dev_files = stream_files(dev_word_ids)

    # Stride in source words; 0 means seq_length.
    words_per_datapoint = dataset_cfg["CONFIG"].get("WORDS_PER_DATAPOINT", 0) or seq_length

    if cfg.get("TRAIN_ALL", False):
        # TRAIN_ALL sets TRAIN_TOKENS=0; size by source words.
        n_src = PosTransitionDataset.source_words(train_files)
        train_sequences = max(0, (n_src - seq_length) // words_per_datapoint)
        print(f"TRAIN_ALL: single pass over {n_src:,} source words "
              f"(stride {words_per_datapoint}) -> {train_sequences:,} examples")
    else:
        train_sequences = cfg["TRAIN_TOKENS"] // seq_length
    dev_sequences = cfg["DEV_TOKENS"] // seq_length

    schedule_cfg = cfg.get("TRANSITION_SCHEDULE", {})
    total_steps = max(1, train_sequences // batch_size)
    words_per_step = batch_size * words_per_datapoint
    schedule_fn = make_schedule_fn(schedule_cfg, total_steps, words_per_step=words_per_step)

    train_dataset = PosTransitionDataset(
        train_files, seq_length, words_per_datapoint,
        schedule_fn=schedule_fn, base_seed=random_seed, num_examples=train_sequences)
    assert len(train_dataset) >= train_sequences, (
        f"Not enough training data: {len(train_dataset)} sequences "
        f"available, {train_sequences} requested")

    eval_dataset = {
        name: PosTransitionDataset(
            dev_files, seq_length, words_per_datapoint,
            fixed_prob_word=prob, base_seed=random_seed, num_examples=dev_sequences)
        for name, prob in (("pos", 0.0), ("word", 1.0))
    }
    for k, ds in eval_dataset.items():
        assert len(ds) >= dev_sequences, (
            f"Not enough dev data for {k!r}: {len(ds)} sequences, "
            f"{dev_sequences} requested")

    print(f"Train: {len(train_dataset):,} examples, seq_length={seq_length}, "
          f"words_per_datapoint={words_per_datapoint}, total_steps={total_steps}")
    print(f"Dev pos/word: {len(eval_dataset['pos']):,} / "
          f"{len(eval_dataset['word']):,} examples")

    info = {"transition_schedule": schedule_cfg, "words_per_datapoint": words_per_datapoint,
            "train_files": train_files, "dev_files": dev_files}
    return train_dataset, eval_dataset, info
