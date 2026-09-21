"""The experiment configuration DSL: every config declares
EXPERIMENTS = {name: {stage_A: [...], stage_B: [...]}}; SConstruct takes the cartesian product."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "common"))
from tags import (  # noqa: E402,F401
    PTB_NON_PUNCT_TAGS, PTB_PUNCT_TAGS, UPOS_NON_PUNCT_TAGS, UPOS_PUNCT_TAGS,
    pos_tag_extras, pos_tag_extras_for,
)


def model_config(architecture_like=None, use_pretrained=False, from_hub=None, tokenizer=None, vocab_size=None, slurm=None, training_overrides=None, graft=None, synthetic=None, **kwargs):
    config = {"TYPE" : "custom_model", "CONFIG" : {}, "HUGGINGFACE_CONFIG" : {}}
    if synthetic is not None:
        config["TYPE"] = "synthetic"
        spec = {k: v for k, v in synthetic.items() if k != "name"}
        spec.setdefault("version", "2b")
        config["CONFIG"]["SYNTHETIC"] = spec
        config["CONFIG"]["NAME"] = synthetic.get("name", "synthetic")
        if tokenizer is not None:
            config["CONFIG"]["TOKENIZER"] = tokenizer
        if vocab_size is not None:
            config["CONFIG"]["VOCAB_SIZE"] = vocab_size
    elif graft is not None:
        config["TYPE"] = "graft"
        config["CONFIG"]["GRAFT"] = {
            "donor": graft.get("donor"),
            "arch": graft.get("arch"),
            "heads": [list(x) for x in graft.get("heads", [])],
            "layers": list(graft.get("layers", [])),
            "mlps": list(graft.get("mlps", [])),
            "body": bool(graft.get("body", False)),
            "with_layernorm": bool(graft.get("with_layernorm", False)),
            "attention_bias": bool(graft.get("attention_bias", False)),
        }
        config["CONFIG"]["NAME"] = graft.get("name", "graft")
        if tokenizer is not None:
            config["CONFIG"]["TOKENIZER"] = tokenizer
        if vocab_size is not None:
            config["CONFIG"]["VOCAB_SIZE"] = vocab_size
    elif from_hub is not None:
        config["TYPE"] = "from_hub"
        config["CONFIG"]["MODEL_NAME"] = from_hub
        config["CONFIG"]["TOKENIZER"] = from_hub if tokenizer is None else tokenizer
        if vocab_size is not None:
            config["CONFIG"]["VOCAB_SIZE"] = vocab_size
    elif architecture_like is not None:
        config["TYPE"] = "architecture_like"
        config["CONFIG"]["ARCHITECTURE_LIKE"] = architecture_like
        config["CONFIG"]["TOKENIZER"] = architecture_like if tokenizer is None else tokenizer
        if vocab_size is not None:
            config["CONFIG"]["VOCAB_SIZE"] = vocab_size
        # Separate body init per seed.
        if kwargs.get("seeded_init"):
            config["CONFIG"]["SEEDED_INIT"] = True
    elif use_pretrained:
        config["TYPE"] = "use_pretrained"
        if tokenizer is not None:
            config["CONFIG"]["TOKENIZER"] = tokenizer
        if vocab_size is not None:
            config["CONFIG"]["VOCAB_SIZE"] = vocab_size
    if slurm is not None:
        config["CONFIG"]["SLURM"] = slurm
    if training_overrides is not None:
        config["CONFIG"]["TRAINING_OVERRIDES"] = training_overrides
    return config

def transition_config(name="none", **kwargs):
    return {"TYPE" : name, "CONFIG" : kwargs}

def dataset_config(name, words=0, **kwargs):
    splits = {
        k.replace("_words","") : v for k, v in kwargs.items()
        if k.endswith("_words") and not k.startswith("source_")
    }
    assert words >= sum(splits.values()), f"Total words must be at least the sum of all splits. Got total: {words}, splits sum: {sum(splits.values())} from {' , '.join([f'{k}:{v}' for k,v in splits.items()])}"
    config = {"CONFIG" : {}, "WORDS" : words, "SPLITS" : splits}
    config["TYPE"] = name
    # Optional keys are emitted only when set, so existing JSON stays byte-identical.
    if "train_split" in kwargs:
        train_key = kwargs["train_split"]
        assert train_key in splits, f"train_split={train_key!r} but splits has keys {list(splits.keys())}"
        config["TRAIN_SPLIT"] = train_key
    if "dev_split" in kwargs:
        dev_key = kwargs["dev_split"]
        assert dev_key in splits, f"dev_split={dev_key!r} but splits has keys {list(splits.keys())}"
        config["DEV_SPLIT"] = dev_key
    # Train on the first N words only; the tokenizer sees the same prefix.
    if "train_prefix" in kwargs:
        prefix = int(kwargs["train_prefix"])
        assert prefix > 0, f"train_prefix must be positive, got {prefix}"
        config["TRAIN_PREFIX"] = prefix
    # Honoured by the aligned pos_transition builder only.
    if "dev_prefix" in kwargs:
        dev_prefix = int(kwargs["dev_prefix"])
        assert dev_prefix > 0, f"dev_prefix must be positive, got {dev_prefix}"
        config["DEV_PREFIX"] = dev_prefix
    # Which split tokenizer="train" trains on.
    if "tokenizer_split" in kwargs:
        tok_key = kwargs["tokenizer_split"]
        assert tok_key in splits, f"tokenizer_split={tok_key!r} but splits has keys {list(splits.keys())}"
        config["TOKENIZER_SPLIT"] = tok_key
    if name == "shuffle_dyck":
        num_symbols = kwargs.get("num_symbols", 64)
        target_length = kwargs.get("target_length", 256)
        p = kwargs.get("p", 0.5)

        default_n = words//target_length
        if words % target_length != 0:
            default_n += 1
        n = kwargs.get("n", default_n)
        config["CONFIG"] = {"NUM_SYMBOLS" : num_symbols, "TARGET_LENGTH" : target_length, "P" : p, "N" : n}
    elif name == "dyck":
        num_symbols = kwargs.get("num_symbols", 64)
        target_length = kwargs.get("target_length", 256)

        default_n = words//target_length
        if words % target_length != 0:
            default_n += 1
        n = kwargs.get("n", default_n)
        min_depth = kwargs.get("min_depth", 1)
        max_depth = kwargs.get("max_depth", 16)
        config["CONFIG"] = {"NUM_SYMBOLS" : num_symbols, "TARGET_LENGTH" : target_length, "N" : n, "MIN_DEPTH" : min_depth, "MAX_DEPTH" : max_depth}   
    elif name == "none":
        config["CONFIG"] = {"NUM_SYMBOLS" : 1, "TARGET_LENGTH" : 1, "N" : 0, "MIN_DEPTH" : 0, "MAX_DEPTH" : 16} 
    elif name in ["pos", "pos_simple", "pos_text"]:
        config["CONFIG"] = {"TREEBANK" : "/export/data/linguistics/penn_treebank/LDC99T42/treebank_3/parsed/mrg/"}
    elif name == "language":
        language = kwargs.get("language", "deu")
        words_per_line_flag = ""
        if "words_per_line" in kwargs:
            words_per_line_flag = f"--words_per_line {kwargs['words_per_line']}"
        config["CONFIG"] = {"LANGUAGE" : language, "WORDS" : words, "WORDS_PER_LINE_FLAG" : words_per_line_flag}
    elif name == "c4":
        c4_config = kwargs.get("c4_config", "en")
        words_per_line_flag = ""
        if "words_per_line" in kwargs:
            words_per_line_flag = f"--words_per_line {kwargs['words_per_line']}"
        config["CONFIG"] = {"C4_CONFIG": c4_config, "WORDS": words, "WORDS_PER_LINE_FLAG": words_per_line_flag}
    elif name == "babylm":
        # Train and dev live in separate HF repos; words=0 takes everything.
        repo = kwargs.get("repo", "BabyLM-community/BabyLM-2026-Strict-Small")
        babylm_config = kwargs.get("babylm_config", "")
        dev_repo = kwargs.get("dev_repo", "BabyLM-community/BabyLM-dev")
        dev_config = kwargs.get("dev_config", "")
        dev_split = kwargs.get("dev_split", "dev")
        config["CONFIG"] = {
            "BABYLM_REPO": repo,
            "BABYLM_CONFIG": babylm_config,
            "BABYLM_DEV_REPO": dev_repo,
            "BABYLM_DEV_CONFIG": dev_config,
            "BABYLM_DEV_SPLIT": dev_split,
            "WORDS": words,
        }
    elif name in ["bnc_tags", "bnc_text"]:
        config["CONFIG"] = {"BNC" : "/home/efittsc1/cleanup/cleanup_pre_pretraining/download/Texts"}
    elif name == "text_file":
        file_path = kwargs.get("file_path")
        file_name = kwargs.get("file_name", os.path.splitext(os.path.basename(file_path))[0])
        config["CONFIG"] = {"FILE_PATH" : file_path, "FILE_NAME" : file_name}
    elif name == "pos_tag":
        # source= wraps any corpus; language= is the legacy path.
        spacy_model = kwargs.get("spacy_model", "en_core_web_sm")
        tag_source = kwargs.get("tag_source", "upos")
        words_per_line_flag = ""
        if "words_per_line" in kwargs:
            words_per_line_flag = f"--words_per_line {kwargs['words_per_line']}"
        source = kwargs.get("source")
        if source is not None:
            config = {"CONFIG": {
                "SOURCE": source,
                "SPACY_MODEL": spacy_model,
                "TAG_SOURCE": tag_source,
                "WORDS_PER_LINE_FLAG": words_per_line_flag,
            },
                "WORDS": source["WORDS"],
                "SPLITS": dict(source["SPLITS"]),
                "TYPE": name}
            if "TRAIN_SPLIT" in source:
                config["TRAIN_SPLIT"] = source["TRAIN_SPLIT"]
            if "DEV_SPLIT" in source:
                config["DEV_SPLIT"] = source["DEV_SPLIT"]
            return config
        language = kwargs.get("language", "eng_Latn")
        config["CONFIG"] = {"LANGUAGE" : language, "WORDS" : words, "SPACY_MODEL" : spacy_model, "TAG_SOURCE" : tag_source, "WORDS_PER_LINE_FLAG" : words_per_line_flag}
    elif name == "pos_tag_text":
        # Tag stream as plain text; split_map maps our splits to source splits.
        source = kwargs["source"]
        split_map = kwargs["split_map"]
        assert set(split_map.keys()) == set(splits.keys()), \
            f"split_map keys {sorted(split_map)} must match declared splits {sorted(splits)}"
        config["CONFIG"] = {
            "SOURCE": source,
            "SPLIT_MAP": split_map,
            "TAG_SOURCE": kwargs.get("tag_source", "ptb"),
            "SPACY_MODEL": kwargs.get("spacy_model", "en_core_web_sm"),
        }
    elif name == "pos_transition":
        # Four CSR memmaps per split; same two source modes as pos_tag.
        spacy_model = kwargs.get("spacy_model", "en_core_web_sm")
        tag_source = kwargs.get("tag_source", "upos")
        words_per_datapoint = kwargs.get("words_per_datapoint", 0)
        # Word side is the exact natural tokenization; aligned=False is refused.
        aligned = kwargs.get("aligned", True)
        words_per_line_flag = ""
        if "words_per_line" in kwargs:
            words_per_line_flag = f"--words_per_line {kwargs['words_per_line']}"
        source = kwargs.get("source")
        if source is not None:
            config = {"CONFIG": {
                "SOURCE": source,
                "SPACY_MODEL": spacy_model,
                "TAG_SOURCE": tag_source,
                "WORDS_PER_DATAPOINT": words_per_datapoint,
                "WORDS_PER_LINE_FLAG": words_per_line_flag,
                "ALIGNED": aligned,
            },
                "WORDS": source["WORDS"],
                "SPLITS": dict(source["SPLITS"]),
                "TYPE": name}
            if "TRAIN_SPLIT" in source:
                config["TRAIN_SPLIT"] = source["TRAIN_SPLIT"]
            if "DEV_SPLIT" in source:
                config["DEV_SPLIT"] = source["DEV_SPLIT"]
            return config
        language = kwargs.get("language", "eng_Latn")
        config["CONFIG"] = {"LANGUAGE" : language, "WORDS" : words, "SPACY_MODEL" : spacy_model, "TAG_SOURCE" : tag_source, "WORDS_PER_DATAPOINT" : words_per_datapoint, "WORDS_PER_LINE_FLAG" : words_per_line_flag, "ALIGNED" : aligned}
    elif name == "ngram":
        source_words = kwargs.get("source_words", 2_000_000_000)
        # source_<split>_words describe the underlying language dataset's splits.
        source_splits = {
            k[len("source_"):-len("_words")]: v
            for k, v in kwargs.items()
            if k.startswith("source_") and k.endswith("_words") and k != "source_words"
        }
        if not source_splits:
            source_splits = {
                "train": source_words - 200_000_000,
                "dev": 100_000_000,
                "test": 100_000_000,
            }
        source_split = kwargs.get("source_split", "train")
        assert source_split in source_splits, (
            f"source_split={source_split!r} but source_splits has keys {list(source_splits.keys())}"
        )
        config["CONFIG"] = {
            "N": kwargs["n"],
            "SOURCE_LANGUAGE": kwargs.get("source_language", "eng_Latn"),
            "SOURCE_WORDS": source_words,
            "SOURCE_SPLITS": source_splits,
            "SOURCE_SPLIT": source_split,
            "SPACY_MODEL": kwargs.get("spacy_model", "en_core_web_sm"),
            "VOCAB_CAP": kwargs.get("vocab_cap", 100_000),
            "WORDS_PER_LINE": kwargs.get("words_per_line", 200),
            "WORDS": words,
        }
    return config

def scheduler_config(name, **kwargs):
    config = {"TYPE" : name, "CONFIG" : {}, "HUGGINGFACE_CONFIG" : {}}
    config["CONFIG"]["decay_steps"] = kwargs.get("decay_steps", 0)
    if "warmup_ratio" in kwargs:
        # HF ignores warmup_ratio if warmup_steps > 0.
        config["HUGGINGFACE_CONFIG"]["warmup_ratio"] = kwargs["warmup_ratio"]
        config["HUGGINGFACE_CONFIG"]["warmup_steps"] = 0
    else:
        config["HUGGINGFACE_CONFIG"]["warmup_steps"] = kwargs.get("warmup_steps", 600)
    if name == "warmup_cosine":
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_type"] = "cosine"
        if "min_lr_rate" in kwargs:
            # Decays to lr * min_lr_rate instead of 0.
            config["HUGGINGFACE_CONFIG"]["lr_scheduler_type"] = "cosine_with_min_lr"
            config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"] = {"min_lr_rate": kwargs["min_lr_rate"]}
    elif name == "wsd":
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_type"] = "constant_with_warmup"
    elif name == "reduce_lr_on_plateau":
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_type"] = "reduce_lr_on_plateau"
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"] = {}
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["factor"] = kwargs.get("factor", 0.5)
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["patience"] = kwargs.get("patience", 4)
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["threshold"] = kwargs.get("threshold", 1e-3)
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["threshold_mode"] = kwargs.get("threshold_mode", "rel")
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["cooldown"] = kwargs.get("cooldown", 2)
        config["HUGGINGFACE_CONFIG"]["lr_scheduler_kwargs"]["min_lr"] = kwargs.get("min_lr", 1e-8)
    return config

def training_config(total_train=5000, total_dev=5000, checkpoint_interval=None, eval_interval=None, return_checkpoints=False, sequence_length=256, batch_size=1, gpus="0", use_bos=False, freeze_layers=False, unfrozen_top=0, unfrozen_bottom=0, freeze_final_norm=None, transition_schedule=None, model_load=None, scale_grid=False, log_eval_steps=None, probe_steps=None, rescale_std=None, **kwargs):
    # sbatch-only: never part of the HuggingFace section or the run's identity.
    slurm_time = kwargs.pop("slurm_time", None)
    slurm_memory = kwargs.pop("slurm_memory", None)
    # total_train="all": one pass over every window; only checkpoint-final is saved.
    train_all = (total_train == "all")
    if train_all:
        assert not checkpoint_interval and not eval_interval, \
            "total_train='all' saves only checkpoint-final; checkpoint_interval/eval_interval must be empty"
        kwargs["save_strategy"] = "custom"
        total_train = 0
    # int: every n tokens; list: at those token counts.
    checkpoint_list = []
    if type(checkpoint_interval) == int:
        if "save_strategy" in kwargs and kwargs["save_strategy"] != "custom":
            print(f"Warning: Overriding save_strategy {kwargs['save_strategy']} to 'custom' due to checkpoint_interval being set.")
        kwargs["save_strategy"] = "custom"
        checkpoint_list = list(range(checkpoint_interval, total_train + 1, checkpoint_interval))
    elif type(checkpoint_interval) == list:
        checkpoint_list = checkpoint_interval
        if "save_strategy" in kwargs and kwargs["save_strategy"] != "custom":
            print(f"Warning: Overriding save_strategy {kwargs['save_strategy']} to 'custom' due to checkpoint_interval being set.")
        kwargs["save_strategy"] = "custom"

    eval_list = []
    if kwargs.get("save_strategy") == "custom":
        if type(eval_interval) == int:
            eval_list = list(range(eval_interval, total_train + 1, eval_interval))
        elif type(eval_interval) == list:
            eval_list = eval_interval
    
    # Round to whole batches.
    batch_tokens = batch_size * sequence_length
    requested_train_tokens = total_train  # names the run path; TRAIN_TOKENS below is the fitted value
    actual_training_tokens = int(total_train // batch_tokens) * batch_tokens
    actual_dev_tokens = int(total_dev // batch_tokens) * batch_tokens
    # Unrounded total, so the final-epoch checkpoint survives rounding.
    multi_epoch_cap = int(total_train * int(kwargs.get("num_train_epochs", 1)))
    actual_checkpoints = [int(cp//batch_tokens)*batch_tokens for cp in checkpoint_list if (cp//batch_tokens)*batch_tokens <= multi_epoch_cap]
    actual_checkpoint = list(sorted(list(set(actual_checkpoints))))
    checkpoint_steps = [int(cp // batch_tokens) for cp in actual_checkpoints]

    actual_evaluations = [int(ev//batch_tokens)*batch_tokens for ev in eval_list if (ev//batch_tokens)*batch_tokens <= multi_epoch_cap]
    actual_evaluations = list(sorted(list(set(actual_evaluations))))
    evaluation_steps = [int(ev // batch_tokens) for ev in actual_evaluations]

    if actual_training_tokens != total_train:
        print(f"Adjusting total training tokens from {total_train} to {actual_training_tokens} to fit batch size and sequence length.")
        total_train = actual_training_tokens
    if actual_dev_tokens != total_dev:
        print(f"Adjusting total dev tokens from {total_dev} to {actual_dev_tokens} to fit batch size and sequence length.")
        total_dev = actual_dev_tokens
    if actual_checkpoints != checkpoint_list:
        print(f"Adjusting checkpoint list from {checkpoint_list} to {actual_checkpoints} to fit batch size and sequence length.")
        checkpoint_list = actual_checkpoints
    if actual_evaluations != eval_list:
        print(f"Adjusting evaluation list from {eval_list} to {actual_evaluations} to fit batch size and sequence length.")
        eval_list = evaluation_steps

    # Explicit optimizer-step list; works in TRAIN_ALL mode.
    if log_eval_steps is not None:
        evaluation_steps = [int(s) for s in log_eval_steps]

    # Checkpoint + probe eval at these steps.
    if probe_steps is not None:
        evaluation_steps = [int(s) for s in probe_steps]
        checkpoint_steps = [int(s) for s in probe_steps]

    device_num = len(gpus.split(","))
    batch_size_per_device = batch_size // device_num
    assert batch_size_per_device * device_num == batch_size, f"Batch size not divisible by number of devices {batch_size} {device_num}, {batch_size_per_device}"
    if "gradient_accumulation_steps" in kwargs:
        gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        batch_size_per_device = batch_size_per_device // gradient_accumulation_steps
        assert batch_size_per_device * gradient_accumulation_steps * device_num == batch_size, f"Batch size not divisible by number of devices and gradient accumulation steps {batch_size} {device_num}, {batch_size_per_device}, {gradient_accumulation_steps}"
    if "per_device_train_batch_size" in kwargs and kwargs["per_device_train_batch_size"] != batch_size_per_device:
        print(f"Warning: Overriding per_device_train_batch_size {kwargs['per_device_train_batch_size']} to {batch_size_per_device} due to batch_size and gpus settings.")
    kwargs["per_device_train_batch_size"] = int(batch_size_per_device)
    kwargs["per_device_eval_batch_size"] = int(batch_size_per_device)

    config_dict = {
                "TYPE" : "standard",
                "CONFIG" : {
                                "CHECKPOINT_LIST_TOKENS" : checkpoint_list,
                                "CHECKPOINT_LIST_STEPS" : checkpoint_steps,
                                "EVALUATION_LIST_STEPS" : evaluation_steps,
                                "RETURN_CHECKPOINTS" : return_checkpoints,
                                "SEQUENCE_LENGTH" : sequence_length,
                                "TRAIN_TOKENS" : total_train,
                                "TRAIN_TOKENS_REQUESTED" : requested_train_tokens,
                                "DEV_TOKENS" : total_dev,
                                "BATCH_SIZE" : batch_size,
                                "GPUS" : gpus,
                                "USE_BOS" : use_bos,
                                "FREEZE_LAYERS" : freeze_layers,
                                "UNFROZEN_TOP" : unfrozen_top,
                                "UNFROZEN_BOTTOM" : unfrozen_bottom,
                            },
                "HUGGINGFACE_CONFIG" : {
                                "eval_strategy" : kwargs.get("eval_strategy", "steps"),
                                "logging_strategy" : kwargs.get("logging_strategy", "steps"),
                                "logging_steps" : kwargs.get("logging_steps", 100),
                                "save_strategy" : kwargs.get("save_strategy", "steps"),
                                "learning_rate" : kwargs.get("lr", 5e-5),
                                "weight_decay" : kwargs.get("wd", 0.01),
                                "num_train_epochs" : kwargs.get("num_train_epochs", 1),
                                "fp16" : kwargs.get("fp16", True),
                                "load_best_model_at_end" : kwargs.get("load_best_model_at_end", True),
                                "metric_for_best_model" : kwargs.get("metric_for_best_model", "eval_loss"),
                                **kwargs
                            }
            }
    # Optional keys: emitted only when set (byte-identical JSON otherwise); each
    # is a SConstruct dispatch signal.
    if train_all:
        config_dict["CONFIG"]["TRAIN_ALL"] = True
    if log_eval_steps is not None:
        config_dict["CONFIG"]["LOG_EVAL"] = True
    if probe_steps is not None:
        config_dict["CONFIG"]["CKPT_PROBE"] = True
    # Body matrices reparametrized as W = c * W', W' at this std.
    if rescale_std is not None:
        assert train_all, "rescale_std is only implemented for total_train='all'"
        config_dict["CONFIG"]["RESCALE_STD"] = rescale_std
    if transition_schedule is not None:
        config_dict["CONFIG"]["TRANSITION_SCHEDULE"] = transition_schedule
    # Adds a /u{unique}K_x{epochs} path segment.
    if scale_grid:
        config_dict["CONFIG"]["SCALE_GRID"] = True
    if model_load is not None:
        config_dict["CONFIG"]["MODEL_LOAD"] = model_load
    if freeze_final_norm is not None:
        config_dict["CONFIG"]["FREEZE_FINAL_NORM"] = bool(freeze_final_norm)
    # Dropped from training_config.json by the SConstruct, so changing them never retrains.
    if slurm_time is not None:
        config_dict["CONFIG"]["SLURM_TIME"] = slurm_time
    if slurm_memory is not None:
        config_dict["CONFIG"]["SLURM_MEMORY"] = slurm_memory
    return config_dict

def eval_config(name, all_checkpoints=False, slurm=None, **kwargs):
    config = {
        "TYPE": name,
        "CONFIG": {"ALL_CHECKPOINTS": all_checkpoints, **kwargs},
    }
    if slurm is not None:
        config["CONFIG"]["SLURM"] = slurm
    return config

# ---------------------------------------------------------------------------
# Tokenizer-extension helpers
# ---------------------------------------------------------------------------

def _dyck_bracket_extras(num_symbols):
    extras = []
    for i in range(2 * num_symbols):
        extras.append(str(i))
        extras.append(" " + str(i))
    return extras


def tokenizer_extended(base, extra_tokens):
    return {"EXTEND_BASE": base, "EXTRA_TOKENS": list(extra_tokens)}


def tokenizer_permuted(base, seed=None):
    return {"PERMUTE_BASE": base, "PERMUTE_SEED": seed}


def tokenizer_with_pos_tags(base, tag_source="ptb", marker=False):
    return tokenizer_extended(base, pos_tag_extras_for(tag_source, marker=marker))


def tokenizer_with_pos_tags_and_dyck(base, tag_source="ptb", num_dyck_symbols=128):
    return tokenizer_extended(
        base,
        pos_tag_extras_for(tag_source) + _dyck_bracket_extras(num_dyck_symbols),
    )
