"""SLURM task settings for one job, read from the site values on the env."""


def task_config(env, name, time_required, memory_required=None, gpu=False):
    if memory_required is None:
        memory_required = env["MEMORY"]
    cfg = {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"] if gpu else env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"] if gpu else env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_NODELIST": env.get("NODELIST", []),
        "STEAMROLLER_EXCLUDE": env.get("EXCLUDE", ""),
    }
    if gpu:
        cfg["STEAMROLLER_GPU_COUNT"] = env["GPU_COUNT"]
    return cfg


def cpu_task_config(env, name, time_required, memory_required=None):
    return task_config(env, name, time_required, memory_required, gpu=False)


def gpu_task_config(env, name, time_required, memory_required=None):
    return task_config(env, name, time_required, memory_required, gpu=True)
