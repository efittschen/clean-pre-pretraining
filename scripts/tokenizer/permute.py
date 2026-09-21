"""Permute a BPE tokenizer's string->id map: same segmentation, same vocab
size, specials pinned; the reset_embeddings leak control."""

import argparse
import json
import os
import pathlib
import shutil

import numpy as np


def _derangement(n, rng):
    perm = [int(x) for x in rng.permutation(n)]
    for i in range(n):
        if perm[i] == i:
            j = int(rng.randint(n))
            while j == i or perm[j] == i:
                j = int(rng.randint(n))
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n)), "derangement fix-up failed"
    return perm


VERIFY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test of the tokenizer permutation.",
    " NNP VBD DT NN . NNS JJ IN PRP$",
    "255 12 3 128 0 1 254",
    "Mixed: naive cafe 3.14159 -- don't; \"quoted\" (parens) [brackets]",
    "a" * 200,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer", required=True,
                        help="Directory of the tokenizer whose ids get permuted")
    parser.add_argument("--output", required=True, help="Output tokenizer directory")
    parser.add_argument("--seed", type=int, required=True,
                        help="RNG seed for the permutation (one permutation per seed)")
    args, _ = parser.parse_known_args()

    src = pathlib.Path(args.input_tokenizer)
    tok_json_path = src / "tokenizer.json"
    if not tok_json_path.exists():
        raise SystemExit(f"[tokenizer_permute] no tokenizer.json in {src}")

    spec = json.loads(tok_json_path.read_text())
    model = spec["model"]
    if model["type"] != "BPE":
        raise SystemExit(
            f"[tokenizer_permute] only BPE tokenizers are supported (got "
            f"{model['type']!r}); Unigram/WordPiece store the vocab as an "
            f"id-ordered list and need a different rewrite."
        )

    vocab = dict(model["vocab"])                # token -> id
    added = spec.get("added_tokens", [])
    v0 = max(vocab.values()) + 1
    if set(vocab.values()) != set(range(v0)):
        raise SystemExit(f"[tokenizer_permute] BPE vocab ids are not 0..{v0 - 1}")

    # Extras get ids by list position at load time.
    in_vocab = [a for a in added if a["content"] in vocab]
    extras = [a for a in added if a["content"] not in vocab]
    extra_ids = {a["id"] for a in extras}
    if extra_ids and extra_ids != set(range(v0, v0 + len(extras))):
        raise SystemExit(
            f"[tokenizer_permute] extra added-token ids are not a contiguous block "
            f"above the BPE vocab ({v0}..{v0 + len(extras) - 1})"
        )

    # ---- ids held fixed -----------------------------------------------------
    fixed = {a["id"] for a in added if a.get("special")}
    unk = model.get("unk_token")
    if unk is not None and unk in vocab:
        fixed.add(vocab[unk])
    fixed_in_base = {i for i in fixed if i < v0}
    # Specials in the added block are pinned by position.
    fixed_extra_pos = {i - v0 for i in fixed if i >= v0}

    rng = np.random.RandomState(args.seed)

    # ---- permute the BPE vocab ---------------------------------------------
    movable = sorted(i for i in range(v0) if i not in fixed_in_base)
    shuffled = [movable[k] for k in _derangement(len(movable), rng)]
    perm = list(range(v0 + len(extras)))        # old -> new
    for old, new in zip(movable, shuffled):
        perm[old] = new

    # ---- extras: permute by reordering the list ----------------------------
    movable_pos = [i for i in range(len(extras)) if i not in fixed_extra_pos]
    shuffled_pos = [movable_pos[k] for k in _derangement(len(movable_pos), rng)]
    pos_map = dict(zip(movable_pos, shuffled_pos))
    pos_map.update({i: i for i in fixed_extra_pos})
    reordered_extras = [None] * len(extras)
    for old_pos, a in enumerate(extras):
        reordered_extras[pos_map[old_pos]] = a
    for new_pos, a in enumerate(reordered_extras):
        perm[a["id"]] = v0 + new_pos
        a["id"] = v0 + new_pos

    assert sorted(perm) == list(range(len(perm))), "permutation is not a bijection"
    moved = sum(1 for i in range(len(perm)) if perm[i] != i)
    print(f"[tokenizer_permute] vocab={len(perm)} (base {v0} + {len(extras)} added)  "
          f"fixed={sorted(fixed_in_base | {v0 + p for p in fixed_extra_pos})}  moved={moved}")

    # ---- rewrite ------------------------------------------------------------
    model["vocab"] = {tok: perm[old] for tok, old in vocab.items()}
    for a in in_vocab:
        a["id"] = perm[a["id"]]
    spec["added_tokens"] = sorted(in_vocab, key=lambda a: a["id"]) + reordered_extras

    pp = spec.get("post_processor")
    if pp is not None and pp.get("type") not in ("ByteLevel", "Sequence"):
        raise SystemExit(
            f"[tokenizer_permute] post_processor type {pp.get('type')!r} may embed token "
            f"ids; extend this script before permuting such a tokenizer."
        )
    for key in ("padding", "truncation"):
        section = spec.get(key)
        if isinstance(section, dict) and "pad_id" in section:
            section["pad_id"] = perm[section["pad_id"]]

    os.makedirs(args.output, exist_ok=True)
    out = pathlib.Path(args.output)
    (out / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False))
    # Other files reference specials by string.
    for f in src.iterdir():
        if f.is_file() and f.name != "tokenizer.json":
            shutil.copy2(f, out / f.name)

    # ---- verify against what the library actually loads ---------------------
    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained(str(src))
    new_tok = AutoTokenizer.from_pretrained(args.output)
    if len(base_tok) != len(new_tok):
        raise SystemExit(f"[tokenizer_permute] vocab size changed: {len(base_tok)} -> {len(new_tok)}")

    # Derived from the loaded objects, not our intent.
    base_vocab, new_vocab = base_tok.get_vocab(), new_tok.get_vocab()
    if set(base_vocab) != set(new_vocab):
        raise SystemExit("[tokenizer_permute] token-string sets differ after permutation")
    realised = {}
    for tok, old in base_vocab.items():
        realised[old] = new_vocab[tok]
    if sorted(realised) != list(range(len(perm))) or sorted(realised.values()) != list(range(len(perm))):
        raise SystemExit("[tokenizer_permute] realised map is not a bijection over 0..V-1")
    if realised != {i: perm[i] for i in range(len(perm))}:
        differing = [i for i in range(len(perm)) if realised[i] != perm[i]]
        raise SystemExit(
            f"[tokenizer_permute] library re-assigned {len(differing)} ids "
            f"(e.g. {differing[:5]}); written ids are not what gets loaded."
        )
    unmoved = {i for i in range(len(perm)) if realised[i] == i}
    expected_unmoved = fixed_in_base | {v0 + p for p in fixed_extra_pos}
    if unmoved != expected_unmoved:
        raise SystemExit(
            f"[tokenizer_permute] ids left in place: {sorted(unmoved)}; expected exactly "
            f"the pinned specials {sorted(expected_unmoved)}"
        )

    for text in VERIFY_TEXTS:
        a = base_tok(text)["input_ids"]
        b = new_tok(text)["input_ids"]
        if [perm[i] for i in a] != b:
            raise SystemExit(
                f"[tokenizer_permute] round-trip mismatch on {text!r}\n"
                f"  base    : {a}\n  permuted: {b}\n  expected: {[perm[i] for i in a]}"
            )
        if base_tok.convert_ids_to_tokens(a) != new_tok.convert_ids_to_tokens(b):
            raise SystemExit(f"[tokenizer_permute] token strings differ on {text!r}")

    # Record the permutation.
    (out / "permutation.json").write_text(
        json.dumps({"seed": args.seed, "source": str(src), "perm": perm})
    )
    print(f"[tokenizer_permute] verified {len(VERIFY_TEXTS)} texts; saved to {args.output}")


if __name__ == "__main__":
    main()
