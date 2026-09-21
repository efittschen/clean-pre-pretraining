import fire
import json
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

import lm_eval


def eval_lm_harness(
    checkpoint,
    tokenizer_path,
    eval_config,
    output,
):
    with open(eval_config, "r") as f:
        cfg = json.load(f)

    tasks = cfg["CONFIG"].get("tasks", ["blimp"])
    if isinstance(tasks, str):
        tasks = [tasks]

    num_fewshot = cfg["CONFIG"].get("num_fewshot", 0)
    batch_size = cfg["CONFIG"].get("batch_size", "auto")
    device = cfg["CONFIG"].get("device", None)

    model_args = f"pretrained={checkpoint},tokenizer={tokenizer_path}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )

    output_data = {
        "eval_type": "lm_harness",
        "checkpoint": checkpoint,
        "tasks": tasks,
        "results": results["results"],
        "n-shot": results.get("n-shot", {}),
        "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    for task_name, task_results in results["results"].items():
        metrics = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in task_results.items()
            if not k.endswith(",stderr")
        )
        print(f"  {task_name}: {metrics}")


if __name__ == "__main__":
    fire.Fire(eval_lm_harness)
