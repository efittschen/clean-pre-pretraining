"""Site configuration: the one file to edit for a new cluster (accounts,
partitions, memory, per-GPU batch table)."""

# ---------------------------------------------------------------------------
# SECTION 1: site values
# ---------------------------------------------------------------------------

ENGINE = "local"          # "slurm" | "local"
STEAMROLLER_ENGINE = ENGINE
GPU_COUNT = 1
MEMORY = "64GB"

GPU_ACCOUNT = "cmessne4"
CPU_ACCOUNT = "tlippin1"

# grad_accum_for() needs a single partition.
GPU_QUEUE = "l40s"
CPU_QUEUE = "cpu"

# See docs/CLUSTER_NOTES.md.
EXCLUDE = "l06"

NODELIST = []

WORK_DIR = "work"


# ---------------------------------------------------------------------------
# SECTION 2: per-device batch sizes
# ---------------------------------------------------------------------------

_PER_DEVICE_BUDGETS = {
    ("160M", "a100"): 16,   # validated
    ("160M", "h100"): 16,
    ("160M", "l40s"): 16,   # UNVALIDATED
    ("345M", "l40s"):  4,   # UNVALIDATED
    ("345M", "a100"):  8,   # UNVALIDATED
    ("600M", "l40s"):  4,   # validated
    ("1B",   "l40s"):  2,   # validated
}


def grad_accum_for(effective_batch, model_size, partition=None):
    p = (partition if partition is not None else GPU_QUEUE).strip()
    if "," in p:
        raise ValueError(
            f"grad_accum_for: GPU_QUEUE={p!r} names multiple partitions, so the "
            "batch cannot be right-sized. Set GPU_QUEUE to a single partition, "
            "or hardcode gradient_accumulation_steps in the experiment config."
        )
    key = (model_size, p)
    if key not in _PER_DEVICE_BUDGETS:
        raise KeyError(
            f"No validated per-device batch for model_size={model_size!r} on "
            f"partition={p!r}. Add an entry to _PER_DEVICE_BUDGETS in "
            f"site_config.py. Known: {sorted(_PER_DEVICE_BUDGETS)}"
        )
    per_device = _PER_DEVICE_BUDGETS[key]
    if effective_batch % per_device != 0:
        raise ValueError(
            f"effective_batch={effective_batch} is not divisible by "
            f"per_device={per_device} ({model_size} on {p})."
        )
    return effective_batch // per_device
