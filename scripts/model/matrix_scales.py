"""Opt-in fixed per-matrix forward multipliers from <model_dir>/matrix_scales.json,
applied as forward hooks on the named Linears."""
import json, os


def install(model, model_dir):
    path = os.path.join(model_dir, "matrix_scales.json")
    if not os.path.isfile(path):
        parent = os.path.join(os.path.dirname(model_dir.rstrip("/")), "matrix_scales.json")
        if os.path.isfile(parent):
            path = parent
        else:
            return 0
    scales = json.load(open(path))
    mods = dict(model.named_modules())
    n = 0
    for pname, s in scales.items():
        mod = mods[pname.rsplit(".", 1)[0]]  # ...weight -> module
        mod.register_forward_hook(lambda m, inp, out, _s=float(s): out * _s)
        n += 1
    print("[matrix_scales] installed %d forward multipliers from %s" % (n, path), flush=True)
    return n


def copy_into(model_dir, checkpoint_dir):
    src = os.path.join(model_dir, "matrix_scales.json")
    if os.path.isfile(src) and os.path.isdir(checkpoint_dir):
        import shutil
        shutil.copy(src, os.path.join(checkpoint_dir, "matrix_scales.json"))
