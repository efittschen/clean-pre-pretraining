"""Opt-in EMA weight averaging, best-checkpoint keeping and early stopping
(EMA_TAUS_STEPS, EVAL_EVERY_STEPS, PATIENCE_EVALS)."""
import json, math, os, shutil, numpy as np, torch
from transformers import TrainerCallback


class EMACallback(TrainerCallback):
    def __init__(self, model, taus, dev_data, dev_tokens, seq_len, eval_every, patience,
                 output_dir, src_model_dir=None, eval_bs=8, main_tau=None, track_tokens=None):
        self.taus = list(taus)
        self.emas = None                      # lazily built on device at first step
        self.n = 0
        self.dev = np.memmap(dev_data, dtype=np.int64, mode="r")[:(track_tokens or dev_tokens)]
        self.seq = seq_len
        self.eval_bs = eval_bs
        self.eval_every = eval_every
        self.patience = patience
        self.out = output_dir
        self.src = src_model_dir
        self.best = {}
        self.stale = 0
        self.hist = []
        hf = os.path.join(output_dir, "ema_history.json")
        if os.path.exists(hf):
            raise RuntimeError("%s already contains a run (ema_history.json). Refusing to write into it "
                               "- a second job would clobber the history and race the best-checkpoint "
                               "saves. Remove the directory or pick a new output_dir." % output_dir)
        print("[ema] taus %s | eval every %d steps on %.2fM dev tokens | patience %d"
              % (self.taus, eval_every, len(self.dev) / 1e6, patience), flush=True)

    # ---- EMA maintenance -------------------------------------------------
    def _init(self, model):
        self.emas = {t: {k: v.detach().clone().float() for k, v in model.state_dict().items()
                         if v.dtype.is_floating_point} for t in self.taus}

    def on_step_end(self, args, state, control, model=None, **kw):
        if model is None: return
        if self.emas is None: self._init(model)
        self.n += 1
        sd = model.state_dict()
        for t, ema in self.emas.items():
            d = min(1.0 - 1.0 / t, self.n / (self.n + 1.0))   # bias-corrected early on
            for k, v in ema.items():
                v.mul_(d).add_(sd[k].detach().float(), alpha=1.0 - d)
        if self.eval_every and self.n % self.eval_every == 0:
            self._checkpoint(model, state, control)

    # ---- evaluation ------------------------------------------------------
    @torch.no_grad()
    def _ppl(self, model):
        model.eval()
        n = len(self.dev) // self.seq
        tot, cnt = 0.0, 0
        for b in range(0, n, self.eval_bs):
            chunk = np.array(self.dev[b * self.seq:(b + self.eval_bs) * self.seq])
            m = len(chunk) // self.seq
            if m == 0: break
            ids = torch.tensor(chunk[:m * self.seq].reshape(m, self.seq)).to(model.device)
            out = model(ids, labels=ids)
            tot += float(out.loss) * m; cnt += m
        model.train()
        return math.exp(tot / max(cnt, 1))

    def _swap(self, model, ema):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        with torch.no_grad():
            for k, v in ema.items(): sd[k].copy_(v.to(sd[k].dtype))
        return backup

    def _restore(self, model, backup):
        sd = model.state_dict()
        with torch.no_grad():
            for k, v in backup.items(): sd[k].copy_(v)

    def _save(self, model, name):
        d = os.path.join(self.out, name)
        tmp = d + ".%s.tmp" % os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        model.save_pretrained(tmp)
        if self.src:
            f = os.path.join(self.src, "matrix_scales.json")
            if os.path.isfile(f): shutil.copy(f, os.path.join(tmp, "matrix_scales.json"))
        if os.path.isdir(d): shutil.rmtree(d)
        os.rename(tmp, d)

    def _checkpoint(self, model, state, control):
        results = {"raw": self._ppl(model)}
        for t in self.taus:
            backup = self._swap(model, self.emas[t])
            results["ema%d" % t] = self._ppl(model)
            self._restore(model, backup)
        improved = []
        for name, ppl in results.items():
            if name not in self.best or ppl < self.best[name] - 1e-6:
                self.best[name] = ppl
                if name == "raw":
                    self._save(model, "best_raw")
                else:
                    t = int(name[3:])
                    backup = self._swap(model, self.emas[t])
                    self._save(model, "best_" + name)
                    self._restore(model, backup)
                improved.append(name)
        self.stale = 0 if improved else self.stale + 1
        self.hist.append({"step": self.n, **results})
        json.dump({"history": self.hist, "best": self.best}, open(os.path.join(self.out, "ema_history.json"), "w"), indent=1)
        best_now = min(results, key=results.get)
        print("[ema] step %6d | %s | best now %s | improved: %s | stale %d/%d"
              % (self.n, "  ".join("%s %.2f" % (k, v) for k, v in results.items()), best_now,
                 ",".join(improved) or "none", self.stale, self.patience), flush=True)
        if self.stale >= self.patience:
            print("[ema] no variant improved for %d evaluations - stopping." % self.stale, flush=True)
            control.should_training_stop = True

    # ---- final dump ------------------------------------------------------
    def save_all(self, model):
        if self.emas is None: return
        for t, ema in self.emas.items():
            backup = self._swap(model, ema)
            self._save(model, "ema_tau%d" % t)
            self._restore(model, backup)
            print("[ema] wrote ema_tau%d" % t, flush=True)
