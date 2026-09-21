## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. edit ONE file for your cluster: accounts, partitions, batch budgets
site_config.py

# 2. see what would run, without running it
scons EXPERIMENT=configs/demo/00_smoke.py -n

# 3. run it
scons EXPERIMENT=configs/demo/00_smoke.py
```