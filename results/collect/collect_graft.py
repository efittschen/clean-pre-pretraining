"""Collect graft results out of the build tree: Stage-B dev perplexity and
the construction gate per arm."""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st


def parse_run(path):
    m = re.search(r"/graft/([^/]+)/seed(\d+)-transition/", path)
    if not m:
        return None
    arm, host_seed = m.group(1), int(m.group(2))
    wm = re.search(r"/w(\d+)M/", path)
    sm = re.findall(r"/seed(\d+)/", path)
    return {"arm": arm, "words_M": int(wm.group(1)) if wm else None,
            "seed": int(sm[-1]) if sm else host_seed}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="work")
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    rows = []
    pattern = os.path.join(a.work, "models", "graft", "**", "stage_B", "checkpoint-final",
                           "eval_perplexity_dev.json")
    for f in glob.glob(pattern, recursive=True):
        meta = parse_run(f)
        if not meta:
            continue
        meta["perplexity"] = json.load(open(f)).get("perplexity")
        gr = os.path.join(a.work, "models", "graft", meta["arm"],
                          "seed%d" % meta["seed"], "graft_report.json")
        if os.path.isfile(gr):
            g = json.load(open(gr))
            meta["tensors_copied"] = g.get("tensors_copied")
            meta["induction_max"] = g.get("induction_max")
            meta["heads_gt_0.2"] = g.get("n_heads_gt_0.2")
        rows.append(meta)

    if not rows:
        print(f"no graft results under {a.work}/models/graft/")
        return

    rows.sort(key=lambda r: (r["words_M"] or 0, r["arm"], r["seed"]))
    print(f"{'B words':>8}  {'arm':<18} {'n':>2} {'dev ppl (median)':>17}  "
          f"{'copied':>6} {'gate':>6}")
    for T in sorted({r["words_M"] for r in rows if r["words_M"]}):
        for arm in sorted({r["arm"] for r in rows if r["words_M"] == T}):
            g = [r for r in rows if r["words_M"] == T and r["arm"] == arm]
            ppl = [r["perplexity"] for r in g if r.get("perplexity")]
            if not ppl:
                continue
            gate = g[0].get("induction_max")
            print(f"{T:>7}M  {arm:<18} {len(ppl):>2} {st.median(ppl):>17.2f}  "
                  f"{str(g[0].get('tensors_copied','?')):>6} "
                  f"{('%.4f' % gate) if gate is not None else '?':>6}")

    if a.csv:
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
