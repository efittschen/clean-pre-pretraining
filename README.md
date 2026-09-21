## Quick start

Needs Python 3.10 or newer (the ML stack in requirements.txt does not support 3.9).

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip

# GPU machines: install the torch build for your CUDA first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU-only or macOS: skip that line; pip picks a default torch.
pip install -r requirements.txt

# 1. edit ONE file for your cluster: accounts, partitions, batch budgets
site_config.py

# 2. see what would run, without running it
scons EXPERIMENT=configs/demo/00_smoke.py -n

# 3. run it
scons EXPERIMENT=configs/demo/00_smoke.py
```
