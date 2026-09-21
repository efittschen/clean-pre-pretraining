"""The build graph: scons EXPERIMENT=<config> [-n].
Per stage and seed: model, dataset, tokenizer, transition, training, evaluation."""
import copy
import itertools
import json
import os
import os.path
import sys

import SCons.Errors
from steamroller import Environment
from SCons.Node import NodeList

sys.path.insert(0, Dir(".").abspath)
sys.path.insert(0, os.path.join(Dir(".").abspath, "build"))

import site_config
from build.paths import (
    POS_STREAM_KEYS, dataset_path_name, format_lr, format_tokens, get_combinations,
    get_tokenizer_id, pos_tokenizer_dataset_name, pos_transition_pretok_name,
    transition_path_name,
)
from build.steamroller import cpu_task_config, gpu_task_config

# Must precede Environment(): steamroller bakes submit_string at init.
from steamroller.engines.slurm_engine import SlurmEngine as _SlurmEngine

_EXCLUDE_CLAUSE = " ${'--exclude=' + STEAMROLLER_EXCLUDE if STEAMROLLER_EXCLUDE else ''}"
if _EXCLUDE_CLAUSE not in _SlurmEngine.submit_string:
    _SlurmEngine.submit_string = _SlurmEngine.submit_string + _EXCLUDE_CLAUSE


class ModelRef:

    def __init__(self, path, flag=None, slurm=None):
        self.path = path
        self.flag = flag
        self.slurm = slurm or {}

    @property
    def abspath(self):
        if isinstance(self.path, NodeList):
            return self.path[0].abspath
        return self.path.abspath

    def as_source(self):
        if self.flag is not None:
            return [self.flag]
        return [self.path]


AddOption("--experiment", dest="experiment", type="string", nargs=1,
          action="store", help="path to the config file to build")

_experiment_file = GetOption("experiment") or ARGUMENTS.get("EXPERIMENT")
if not _experiment_file:
    raise SCons.Errors.UserError(
        "No experiment selected.\n"
        "  scons EXPERIMENT=configs/demo/00_smoke.py\n"
        "Available:\n    " + "\n    ".join(sorted(
            os.path.join(dp, f) for dp, _, fs in os.walk("configs")
            for f in fs if f.endswith(".py"))))
if not os.path.exists(_experiment_file):
    raise SCons.Errors.UserError(f"No such experiment config: {_experiment_file}")

vars = Variables(_experiment_file)
vars.AddVariables(
    ("EXPERIMENTS", "", {}),
    ("RANDOM_SEEDS", "", [42]),
    ("STEAMROLLER_ENGINE", "", site_config.ENGINE),
    ("GPU_COUNT", "", site_config.GPU_COUNT),
    ("MEMORY", "", site_config.MEMORY),
    ("GPU_ACCOUNT", "", site_config.GPU_ACCOUNT),
    ("CPU_ACCOUNT", "", site_config.CPU_ACCOUNT),
    ("GPU_QUEUE", "", site_config.GPU_QUEUE),
    ("CPU_QUEUE", "", site_config.CPU_QUEUE),
    ("STEAMROLLER_EXCLUDE", "", site_config.EXCLUDE),
    ("STEAMROLLER_NODELIST", "", site_config.NODELIST),
    ("WORK_DIR", "", site_config.WORK_DIR),
)

print(f"[experiment] {_experiment_file}")


env = Environment(
    variables=vars,
    BUILDERS={
        "GenerateLanguage": Builder(
            action=(
                "python scripts/data/fw_language.py "
                "--language ${LANGUAGE} "
                "--words ${WORDS} "
                "--output ${TARGETS} "
                "${WORDS_PER_LINE_FLAG} "
            )
        ),
        "GenerateC4": Builder(
            action=(
                "python scripts/data/c4.py "
                "--config ${C4_CONFIG} "
                "--words ${WORDS} "
                "--output ${TARGETS} "
                "${WORDS_PER_LINE_FLAG} "
            )
        ),
        "GenerateBabyLM": Builder(
            action=(
                "python scripts/data/babylm.py "
                "--repo ${BABYLM_REPO} "
                "--output ${TARGETS} "
                "${BABYLM_CONFIG_FLAG} "
                "${BABYLM_SPLIT_FLAG} "
                "${WORDS_FLAG} "
            )
        ),
        "GenerateNgram": Builder(
            action=(
                "python scripts/data/ngram.py "
                "--source_tsv ${SOURCE_TSV} "
                "--n ${N} "
                "--target_words ${WORDS} "
                "--output ${TARGETS} "
                "--vocab_cap ${VOCAB_CAP} "
                "--words_per_line ${WORDS_PER_LINE} "
                "--random_seed ${RANDOM_SEED} "
            )
        ),
        "GenerateDyckShuffle": Builder(
            action=(
                "python scripts/data/dyck_shuffle.py "
                "--output ${TARGETS} "
                "--num_symbols ${NUM_SYMBOLS} "
                "--n ${N} "
                "--target_length ${TARGET_LENGTH} "
                "--new_bracket_prob ${P} "
            )
        ),
        "SplitDataset": Builder(
            action=(
                "python scripts/data/split.py "
                "--input ${INPUT} "
                "--output_files ${TARGETS} "
                "--split_sizes ${SPLIT_SIZES} "
                "--seed ${RANDOM_SEED} "
            )
        ),
        "ConcatFiles": Builder(
            action=(
                "python scripts/data/concat.py "
                "--inputs ${SOURCES} "
                "--output ${TARGETS[0]} "
            )
        ),
        "PosTagText": Builder(
            action=(
                "python scripts/data/pos_tag_text.py "
                "--input ${INPUT} "
                "--output ${TARGETS[0]} "
                "--tag_source ${TAG_SOURCE} "
            )
        ),
        "ModelLike": Builder(
            action=(
                "python scripts/model/like.py "
                "--architecture_like ${ARCHITECTURE_LIKE} "
                "--output ${TARGETS} "
                "${SEED_FLAG} "
            )
        ),
        "GetModelFromHub": Builder(
            action=(
                "python scripts/model/from_hub.py "
                "--model_name ${MODEL_NAME} "
                "--output ${TARGETS} "
            )
        ),
        "ResetEmbeddings": Builder(
            action=(
                "python scripts/model/reset_embeddings.py "
                "--input_model ${INPUT_MODEL} "
                "--output ${OUTPUT_DIR} "
                "${TOKENIZER_FLAG} "
                "--std ${STD} "
                "--flag ${TARGETS[0]} "
            )
        ),
        "GetTokenizer": Builder(
            action=(
                "python scripts/tokenizer/get.py "
                "--input ${MODEL_NAME} "
                "--output ${TARGETS} "
            )
        ),
        "TrainTokenizerWords": Builder(
            action=(
                "python scripts/tokenizer/train_words.py "
                "--input ${INPUT} "
                "--output ${TARGETS} "
                "--words ${WORDS} "
                "--vocab_size ${VOCAB_SIZE} "
                "--seq_length ${SEQ_LENGTH} "
            )
        ),
        "ExtendTokenizer": Builder(
            action=(
                "python scripts/tokenizer/extend.py "
                "--base_tokenizer ${BASE_TOKENIZER} "
                "--extra_tokens ${EXTRA_TOKENS} "
                "--output ${TARGETS} "
            )
        ),
        "PermuteTokenizer": Builder(
            action=(
                "python scripts/tokenizer/permute.py "
                "--input_tokenizer ${INPUT_TOKENIZER} "
                "--seed ${PERM_SEED} "
                "--output ${TARGETS} "
            )
        ),
        "TokenizeFile": Builder(
            action=(
                "python scripts/tokenizer/tokenize_file.py "
                "--tokenizer ${TOKENIZER} "
                "--input ${INPUT} "
                "--output ${TARGETS} "
                "${WORDS_FLAG} "
            )
        ),
        "EvalLMHarness": Builder(
            action=(
                "python scripts/eval/lm_harness.py "
                "--checkpoint ${CHECKPOINT} "
                "--tokenizer_path ${TOKENIZER} "
                "--eval_config ${EVAL_CONFIG} "
                "--output ${TARGETS[0]} "
            )
        ),
        "TrainModel": Builder(
            action=(
                "python scripts/model/train.py "
                "--model ${MODEL} "
                "--output_dir ${OUTPUT_DIR} "
                "--train_data ${TRAIN_DATA} "
                "--dev_data ${DEV_DATA} "
                "--training_config ${TRAINING_CONFIG} "
                "--scheduler_config ${SCHEDULER_CONFIG} "
                "--dataset_config ${DATASET_CONFIG} "
                "--tokenizer_path ${TOKENIZER} "
                "--random_seed ${RANDOM_SEED} "
                "--flag ${TARGETS[0]} "
            )
        ),
        "EvalPerplexity": Builder(
            action=(
                "python scripts/eval/perplexity.py "
                "--checkpoint ${CHECKPOINT} "
                "--tokenizer_path ${TOKENIZER} "
                "--eval_config ${EVAL_CONFIG} "
                "--output ${TARGETS[0]} "
                "--stream ${STREAM} --mask ${MASK} "
            )
        ),
        "Graft": Builder(action=(
            "python scripts/transplant/graft.py "
            "--spec ${SPEC} --output ${OUT_DIR} "
            "--random_seed ${RANDOM_SEED} --flag ${TARGETS[0]} "
        )),
        "SyntheticInit": Builder(action=(
            "python scripts/transplant/synthetic_init.py "
            "--spec ${SPEC} --output ${OUT_DIR} "
            "--random_seed ${RANDOM_SEED} --flag ${TARGETS[0]} "
        )),
        "PosPreprocess": Builder(
            action=(
                "python scripts/data/pos_preprocess.py "
                "--input ${INPUT} "
                "--output ${TARGETS[0]} "
                "--spacy_model ${SPACY_MODEL} "
                "--chunk_idx ${CHUNK_IDX} "
                "--num_chunks ${NUM_CHUNKS} "
            )
        ),
        "PosTagTokenize": Builder(
            action=(
                "python scripts/data/pos_tag_tokenize.py "
                "--input ${INPUT} "
                "--tokenizer ${TOKENIZER} "
                "--seq_length ${SEQ_LENGTH} "
                "--output ${TARGETS[0]} "
                "--tag_source ${TAG_SOURCE} "
                "--random_seed ${RANDOM_SEED} "
            )
        ),
        "PosPretokenizeAligned": Builder(
            action=(
                "python scripts/data/pos_pretokenize_aligned.py "
                "--input ${INPUT} "
                "--tokenizer ${TOKENIZER} "
                "--output_word_ids ${TARGETS[0]} "
                "--output_word_offsets ${TARGETS[1]} "
                "--output_pos_ids ${TARGETS[2]} "
                "--output_pos_offsets ${TARGETS[3]} "
                "--tag_source ${TAG_SOURCE} "
                "--spacy_model ${SPACY_MODEL} "
                "${MAX_WORDS_FLAG} "
            )
        ),
    },
)

EVAL_TYPES = {
    "perplexity":                   ("EvalPerplexity", {"STREAM": "tok", "MASK": "none"}),
    "perplexity_word_ids":          ("EvalPerplexity", {"STREAM": "word_ids", "MASK": "none"}),
    "perplexity_word_ids_langonly": ("EvalPerplexity", {"STREAM": "word_ids", "MASK": "grammar_tags"}),
    "perplexity_pos_ids":           ("EvalPerplexity", {"STREAM": "pos_ids", "MASK": "none"}),
    "lm_harness":                   ("EvalLMHarness", {}),
}


# Variables() swallows config exceptions; re-run to surface them.
if not env.get("EXPERIMENTS"):
    try:
        _ns = {"__file__": _experiment_file, "__name__": "__experiment__"}
        exec(compile(open(_experiment_file).read(), _experiment_file, "exec"), _ns)
    except Exception as _e:
        raise SCons.Errors.UserError(
            f"{_experiment_file} failed to load: {type(_e).__name__}: {_e}\n"
            "SCons hid this because Variables() swallows config errors."
        )
    raise SCons.Errors.UserError(
        f"{_experiment_file} loaded but declares no work.\n"
        "A config must define EXPERIMENTS."
    )


def update_config(file, config):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    new_config = json.dumps(config, indent=4, sort_keys=True)
    if os.path.exists(file):
        existing_config_str = json.dumps(json.load(open(file, "r")), indent=4, sort_keys=True)
        if existing_config_str == new_config:
            return env.File(file)
        print(f"Updating config file {file}, because something changed.")
        os.rename(file, file + ".backup")
    json.dump(config, open(file, "w"), indent=4, sort_keys=True)
    return env.File(file)


def checkpoint_dir_names(training_config, dataset_config):
    cfg = training_config["CONFIG"]
    if cfg.get("TRAIN_ALL"):
        return ["checkpoint-final"]
    if dataset_config["WORDS"] == 0:
        return ["checkpoint-0"]
    num_epochs = int(training_config["HUGGINGFACE_CONFIG"].get("num_train_epochs", 1))
    cap = (cfg.get("TRAIN_TOKENS", 0) * max(1, num_epochs)
           + cfg.get("SEQUENCE_LENGTH", 1) * cfg.get("BATCH_SIZE", 1))
    return [f"checkpoint-{step}"
            for step, tokens in zip(cfg.get("CHECKPOINT_LIST_STEPS", []),
                                    cfg.get("CHECKPOINT_LIST_TOKENS", []))
            if tokens <= cap]


# Datasets

SIMPLE_DATASETS = {
    "language":     ("GenerateLanguage",    "06:00:00", "24GB", 0),
    "c4":           ("GenerateC4",          "06:00:00", "24GB", 0),
    "shuffle_dyck": ("GenerateDyckShuffle", "08:00:00", "64GB", 0),
}


def preprocess_language_split(corpus_segment, random_seed, split_file, split_key,
                              spacy_model="en_core_web_sm", num_chunks=16):
    preproc_dir = f"work/datasets/{corpus_segment}/seed{random_seed}/preprocessed/{spacy_model}"
    chunk_nodes = []
    for i in range(num_chunks):
        chunk_target = f"{preproc_dir}/{split_key}.chunk{i}of{num_chunks}.tsv"
        chunk_node = env.PosPreprocess(
            source=[split_file],
            target=[chunk_target],
            INPUT=split_file,
            SPACY_MODEL=spacy_model,
            CHUNK_IDX=i,
            NUM_CHUNKS=num_chunks,
            **cpu_task_config(env, f"pos_preproc_{split_key}_c{i}", "08:00:00", "32GB"),
        )
        chunk_nodes.append(chunk_node[0])
    preproc_target = f"{preproc_dir}/{split_key}.tsv"
    concat_node = env.ConcatFiles(
        source=chunk_nodes,
        target=[preproc_target],
        **cpu_task_config(env, f"pos_concat_{split_key}", "01:00:00", "8GB"),
    )
    return concat_node[0]


def get_dataset(config, random_seed):
    if config["TYPE"] == "pos_tag_text":
        c = config["CONFIG"]
        source_cfg = c["SOURCE"]
        src = get_dataset(source_cfg, random_seed)
        my_name = dataset_path_name(config)[0]
        source_segment = dataset_path_name(source_cfg)[0]
        chunks_per_split = {"train": 16, "trainA": 16, "trainB": 16, "decay": 2, "dev": 2, "test": 2}
        splits = {"dataset": src["dataset"]}
        for our_key, src_key in sorted(c["SPLIT_MAP"].items()):
            tsv = preprocess_language_split(
                source_segment, random_seed, src[src_key], src_key,
                spacy_model=c.get("SPACY_MODEL", "en_core_web_sm"),
                num_chunks=chunks_per_split.get(src_key, 4),
            )
            node = env.PosTagText(
                source=[tsv],
                target=[f"work/datasets/{my_name}/seed{random_seed}/split_{our_key}.txt"],
                INPUT=tsv,
                TAG_SOURCE=c.get("TAG_SOURCE", "ptb"),
                **cpu_task_config(env, f"pos_tag_text_{our_key}", "24:00:00", "16GB"),
            )
            splits[our_key] = node[0]
        return splits

    if config["TYPE"] in ("pos_tag", "pos_transition"):
        source = config["CONFIG"].get("SOURCE")
        if source is not None:
            return get_dataset(source, random_seed)
        underlying = {**config, "TYPE": "language",
                      "CONFIG": {k: v for k, v in config["CONFIG"].items()
                                 if k in ("LANGUAGE", "WORDS", "WORDS_PER_LINE_FLAG")}}
        return get_dataset(underlying, random_seed)

    all_names = dataset_path_name(config, all=True)
    my_name = dataset_path_name(config)[0]

    if config["TYPE"] in SIMPLE_DATASETS:
        dataset_config = SIMPLE_DATASETS.get(config["TYPE"])
        builder_name = dataset_config[0]
        dataset = getattr(env, builder_name)(
            source=[],
            target=[f"work/datasets/{ds}/full.txt" for ds in all_names],
            **config["CONFIG"],
            **cpu_task_config(env, config["TYPE"], dataset_config[1], dataset_config[2]),
        )
        dataset = dataset[dataset_config[3]]
    elif config["TYPE"] == "babylm":
        c = config["CONFIG"]
        splits_dir = f"work/datasets/{my_name}/seed{random_seed}"
        splits_dict = {}
        for split_key, split_words in config["SPLITS"].items():
            if split_key == "dev":
                repo = c.get("BABYLM_DEV_REPO", c["BABYLM_REPO"])
                bl_cfg = c.get("BABYLM_DEV_CONFIG", c["BABYLM_CONFIG"])
                bl_split = c.get("BABYLM_DEV_SPLIT", "dev")
            else:
                repo = c["BABYLM_REPO"]
                bl_cfg = c["BABYLM_CONFIG"]
                bl_split = "train" if split_key == "train" else split_key
            words_flag = f"--words {split_words}" if split_words else ""
            # An empty --config would swallow --split.
            config_flag = f"--config {bl_cfg}" if bl_cfg else ""
            split_flag = f"--split {bl_split}" if bl_split else ""
            out_file = f"{splits_dir}/split_{split_key}.txt"
            result = env.GenerateBabyLM(
                source=[],
                target=[out_file],
                BABYLM_REPO=repo,
                BABYLM_CONFIG_FLAG=config_flag,
                BABYLM_SPLIT_FLAG=split_flag,
                WORDS_FLAG=words_flag,
                **cpu_task_config(env, f"babylm_{split_key}", "02:00:00", "24GB"),
            )
            splits_dict[split_key] = result[0]
        return {"dataset": splits_dict.get("train", next(iter(splits_dict.values()))),
                "dataset_name": my_name, **splits_dict}
    elif config["TYPE"] == "text_file":
        dataset = env.File(config["CONFIG"]["FILE_PATH"])
    elif config["TYPE"] == "ngram":
        c = config["CONFIG"]
        src_lang = c["SOURCE_LANGUAGE"]
        src_words = c["SOURCE_WORDS"]
        n = c["N"]
        spacy_model = c.get("SPACY_MODEL", "en_core_web_sm")
        vocab_cap = c.get("VOCAB_CAP", 100_000)
        words_per_line = c.get("WORDS_PER_LINE", 200)

        src_split = c["SOURCE_SPLIT"]
        src_cfg = copy.deepcopy(config)
        src_cfg["TYPE"] = "language"
        src_cfg["CONFIG"] = {"LANGUAGE": src_lang, "WORDS": src_words, "SPACY_MODEL": spacy_model, "WORDS_PER_LINE_FLAG": ""}
        src_cfg["WORDS"] = src_words
        src_cfg["SPLITS"] = c["SOURCE_SPLITS"]
        src_data = get_dataset(src_cfg, random_seed)

        src_train_tsv = preprocess_language_split(
            dataset_path_name(src_cfg)[0], random_seed, src_data[src_split], src_split,
            spacy_model=spacy_model, num_chunks=16,
        )

        dataset = env.GenerateNgram(
            source=[src_train_tsv],
            target=[f"work/datasets/{my_name}/seed{random_seed}/full.txt"],
            SOURCE_TSV=src_train_tsv,
            N=n,
            WORDS=config["WORDS"],
            VOCAB_CAP=vocab_cap,
            WORDS_PER_LINE=words_per_line,
            RANDOM_SEED=random_seed,
            **cpu_task_config(env, f"ngram_n{n}", "12:00:00", "164GB"),
        )
        dataset = dataset[0]
        my_name = f"{my_name}/seed{random_seed}"
    else:
        raise ValueError(f"Unknown dataset type {config['TYPE']}")

    output_files = []
    for key in config["SPLITS"]:
        output_files.append(f"work/datasets/{my_name}/seed{random_seed}/split_{key}.txt")
    splits = env.SplitDataset(
        source=[dataset],
        target=output_files,
        INPUT=dataset,
        SPLIT_SIZES=config["SPLITS"].values(),
        RANDOM_SEED=random_seed,
        **cpu_task_config(env, "split_dataset", "2:30:00", "64GB"),
    )

    splits = dict(zip(config["SPLITS"].keys(), splits))
    return {"dataset": dataset, "dataset_name": my_name, **splits}


# Tokenizers

def get_tokenizer(model_config, pretrained_tokenizer=None, pretrained_tok_name=None, dataset=None, dataset_name=None, random_seed=None, dataset_splits=None, dataset_config=None):
    config = copy.deepcopy(model_config)["CONFIG"]
    tok_setting = config.get("TOKENIZER")
    tok_split = None
    if dataset_config is not None:
        tok_split = dataset_config.get("TOKENIZER_SPLIT")

    if isinstance(tok_setting, dict) and "PERMUTE_BASE" in tok_setting:
        base_spec = tok_setting["PERMUTE_BASE"]
        perm_seed = tok_setting.get("PERMUTE_SEED")
        if perm_seed is None:
            if random_seed is None:
                raise ValueError(
                    "tokenizer_permuted(seed=None) needs the run's random_seed, "
                    "but get_tokenizer was called without one."
                )
            perm_seed = random_seed
        base_tokenizer, base_tok_name = get_tokenizer(
            {"CONFIG": {**config, "TOKENIZER": base_spec}},
            pretrained_tokenizer=pretrained_tokenizer,
            pretrained_tok_name=pretrained_tok_name,
            dataset=dataset, dataset_name=dataset_name, random_seed=random_seed,
            dataset_splits=dataset_splits, dataset_config=dataset_config,
        )
        tok_name = f"permuted/{base_tok_name.replace('/', '_')}/s{perm_seed}"
        tokenizer = env.PermuteTokenizer(
            source=[base_tokenizer],
            target=Dir(f"work/tokenizers/{tok_name}"),
            INPUT_TOKENIZER=base_tokenizer,
            PERM_SEED=perm_seed,
            **cpu_task_config(env, "permute_tok", "00:30:00", "8GB"),
        )
    elif isinstance(tok_setting, dict) and "EXTEND_BASE" in tok_setting:
        import hashlib
        import urllib.parse
        base = tok_setting["EXTEND_BASE"]
        extras = list(tok_setting["EXTRA_TOKENS"])
        safe_base = base.replace("/", "_")
        encoded_extras = ",".join(urllib.parse.quote(t, safe="") for t in extras)
        extras_hash = hashlib.md5(",".join(sorted(extras)).encode()).hexdigest()[:8]
        if base == "train":
            base_cfg = copy.deepcopy(model_config)
            base_cfg["CONFIG"]["TOKENIZER"] = "train"
            base_tok, base_name = get_tokenizer(
                base_cfg, pretrained_tokenizer, pretrained_tok_name, dataset, dataset_name,
                random_seed, dataset_splits, dataset_config)
            tok_name = f"{base_name}_ext_n{len(extras)}_{extras_hash}"
            tokenizer = env.ExtendTokenizer(
                source=[base_tok],
                target=Dir(f"work/tokenizers/{tok_name}"),
                BASE_TOKENIZER=base_tok[0].abspath,
                EXTRA_TOKENS=encoded_extras,
                **cpu_task_config(env, "extend_tok", "00:30:00", "8GB"),
            )
        else:
            tok_name = f"extended_{safe_base}_n{len(extras)}_{extras_hash}"
            tokenizer = env.ExtendTokenizer(
                source=[],
                target=Dir(f"work/tokenizers/{tok_name}"),
                BASE_TOKENIZER=base,
                EXTRA_TOKENS=encoded_extras,
                **cpu_task_config(env, "extend_tok", "00:30:00", "8GB"),
            )
    elif isinstance(tok_setting, dict):
        tok_ds_config = tok_setting
        tok_ds_name = dataset_path_name(tok_ds_config)[0]
        tok_dataset = get_dataset(tok_ds_config, random_seed)
        vocab_size = config.get("VOCAB_SIZE", 16000)
        seq_length = config.get("SEQUENCE_LENGTH", 128)
        tok_name = f"{tok_ds_name}/v{vocab_size}"
        tokenizer = env.TrainTokenizerWords(
            source=[tok_dataset["dataset"]],
            target=Dir(f"work/tokenizers/{tok_name}"),
            INPUT=tok_dataset["dataset"], WORDS=tok_ds_config["WORDS"],
            VOCAB_SIZE=vocab_size, SEQ_LENGTH=seq_length,
            **cpu_task_config(env, "train_tok", "06:00:00", "24GB"),
        )
    elif tok_setting == "train":
        if dataset is None:
            raise ValueError("dataset must be provided when tokenizer='train'")
        vocab_size = config.get("VOCAB_SIZE", 16000)
        seq_length = config.get("SEQUENCE_LENGTH", 128)
        train_prefix = dataset_config.get("TRAIN_PREFIX") if dataset_config is not None else None
        if tok_split is not None and train_prefix:
            if dataset_splits is None or tok_split not in dataset_splits:
                raise ValueError(
                    f"tokenizer_split={tok_split!r} requested but dataset_splits is "
                    f"{None if dataset_splits is None else list(dataset_splits.keys())}"
                )
            prefix_label = (f"{train_prefix // 1_000_000}Mw"
                            if train_prefix % 1_000_000 == 0 else f"{train_prefix}w")
            tok_input = dataset_splits[tok_split]
            tok_name = f"{dataset_name}/{tok_split}_{prefix_label}/seed{random_seed}/v{vocab_size}"
            tokenizer = env.TrainTokenizerWords(
                source=[tok_input],
                target=Dir(f"work/tokenizers/{tok_name}"),
                INPUT=tok_input, WORDS=train_prefix,
                VOCAB_SIZE=vocab_size, SEQ_LENGTH=seq_length,
                **cpu_task_config(env, "train_tok_words", "06:00:00", "24GB"),
            )
            return tokenizer, tok_name
        if tok_split is not None:
            # Seed-free target on a per-seed split: single-seed only.
            if dataset_splits is None or tok_split not in dataset_splits:
                raise ValueError(
                    f"tokenizer_split={tok_split!r} requested but dataset_splits is "
                    f"{None if dataset_splits is None else list(dataset_splits.keys())}"
                )
            tok_input = dataset_splits[tok_split]
            tok_name = f"{dataset_name}/{tok_split}/v{vocab_size}"
        else:
            tok_input = dataset
            tok_name = f"{dataset_name}/v{vocab_size}"
        if dataset_config is None:
            raise ValueError("tokenizer='train' needs the dataset_config to size the tokenizer's input")
        tok_words = dataset_config["SPLITS"][tok_split] if tok_split is not None else dataset_config["WORDS"]
        tokenizer = env.TrainTokenizerWords(
            source=[tok_input],
            target=Dir(f"work/tokenizers/{tok_name}"),
            INPUT=tok_input, WORDS=tok_words,
            VOCAB_SIZE=vocab_size, SEQ_LENGTH=seq_length,
            **cpu_task_config(env, "train_tok", "06:00:00", "24GB"),
        )
    elif tok_setting is not None:
        safe_tok = tok_setting.replace("/", "_")
        tok_name = f"tokenizer_{safe_tok}"
        tokenizer = env.GetTokenizer(
            source=[],
            target=Dir(f"work/tokenizers/{tok_name}"),
            MODEL_NAME=tok_setting,
            **cpu_task_config(env, "get_tokenizer", "01:00:00", "8GB"),
        )
    elif pretrained_tokenizer is not None:
        return pretrained_tokenizer, pretrained_tok_name
    else:
        raise ValueError("No tokenizer configured and no pretrained_tokenizer provided")

    return tokenizer, tok_name


def get_tokenized_datasets(dataset, tokenizer, dataset_name, tok_name, random_seed, dataset_config=None):
    train_prefix = dataset_config.get("TRAIN_PREFIX") if dataset_config is not None else None
    if train_prefix:
        train_key = dataset_config.get("TRAIN_SPLIT", "train")
        dev_key = dataset_config.get("DEV_SPLIT", "dev")
        tokenized_datasets = {}
        tokenized_datasets[train_key] = env.TokenizeFile(
            source=[dataset[train_key], tokenizer],
            target=f"work/datasets/{dataset_name}/seed{random_seed}/tokenized/{tok_name}/{train_key}.tok",
            INPUT=dataset[train_key],
            TOKENIZER=tokenizer,
            WORDS_FLAG=f"--words {train_prefix}",
            **cpu_task_config(env, f"tokenize_words_{train_key}", "06:00:00", "64GB"),
        )[0]
        tokenized_datasets[dev_key] = env.TokenizeFile(
            source=[dataset[dev_key], tokenizer],
            target=f"work/datasets/{dataset_name}/seed{random_seed}/tokenized/{tok_name}/{dev_key}.tok",
            INPUT=dataset[dev_key],
            TOKENIZER=tokenizer,
            **cpu_task_config(env, f"tokenize_{dev_key}", "06:00:00", "64GB"),
        )[0]
        return tokenized_datasets
    tokenized_datasets = {}
    for key, value in dataset.items():
        if key in ("dataset", "dataset_name"):
            continue
        tokenized_value = env.TokenizeFile(
            source=[value, tokenizer],
            target=f"work/datasets/{dataset_name}/seed{random_seed}/tokenized/{tok_name}/{key}.tok",
            INPUT=value,
            TOKENIZER=tokenizer,
            **cpu_task_config(env, f"tokenize_{key}", "06:00:00", "64GB"),
        )
        tokenized_datasets[key] = tokenized_value[0]
    return tokenized_datasets


# Models

def get_model(config, random_seed, trained=None):
    config = copy.deepcopy(config)
    config["random_seed"] = random_seed
    slurm = config["CONFIG"].get("SLURM", {})

    if config["TYPE"] == "architecture_like":
        arch_basename = os.path.splitext(os.path.basename(config["CONFIG"]["ARCHITECTURE_LIKE"]))[0]
        if config["CONFIG"].get("SEEDED_INIT"):
            file = Dir(f"work/models/model_like_{arch_basename}_seed{random_seed}")
            seed_flag = f"--random_seed {random_seed}"
        else:
            file = Dir(f"work/models/model_like_{arch_basename}")
            seed_flag = ""
        node = env.ModelLike(
            source=[],
            target=file,
            SEED_FLAG=seed_flag,
            ARCHITECTURE_LIKE=config["CONFIG"]["ARCHITECTURE_LIKE"],
            **cpu_task_config(env, "model_like", "02:30:00", "48GB"),
        )
        model = ModelRef(node, slurm=slurm)
    elif config["TYPE"] == "from_hub":
        model_name = config["CONFIG"]["MODEL_NAME"]
        safe_name = model_name.replace("/", "_")
        file = Dir(f"work/models/{safe_name}")
        node = env.GetModelFromHub(
            source=[],
            target=file,
            MODEL_NAME=model_name,
            **cpu_task_config(env, "from_hub", "01:00:00", "48GB"),
        )
        model = ModelRef(node, slurm=slurm)
    elif config["TYPE"] == "synthetic":
        sspec = config["CONFIG"]["SYNTHETIC"]
        name = config["CONFIG"].get("NAME", "synthetic")
        out = f"work/models/synthetic/{name}/seed{random_seed}"
        spec_file = update_config(f"{out}/synthetic_spec.json", sspec)
        node = env.SyntheticInit(
            source=[spec_file],
            target=[f"{out}/synthetic.flag"],
            SPEC=spec_file, OUT_DIR=out, RANDOM_SEED=random_seed,
            **gpu_task_config(env, f"synthetic_{name}", "01:00:00", "32GB"),
        )
        model = ModelRef(Dir(out), node, slurm=slurm)
    elif config["TYPE"] == "graft":
        gspec = copy.deepcopy(config["CONFIG"]["GRAFT"])
        name = config["CONFIG"].get("NAME", "graft")
        out = f"work/models/graft/{name}/seed{random_seed}"

        donor_sources = []
        if isinstance(gspec.get("donor"), dict):
            donor_model = get_model(gspec["donor"], random_seed)
            gspec["donor"] = donor_model.abspath
            donor_sources = donor_model.as_source()

        spec_file = update_config(f"{out}/graft_spec.json", gspec)
        node = env.Graft(
            source=[spec_file] + donor_sources,
            target=[f"{out}/graft.flag"],
            SPEC=spec_file,
            OUT_DIR=out,
            RANDOM_SEED=random_seed,
            **gpu_task_config(env, f"graft_{name}", "01:00:00", "32GB"),
        )
        model = ModelRef(Dir(out), node, slurm=slurm)
    elif config["TYPE"] == "use_pretrained":
        if trained is None:
            raise ValueError("a trained model must be provided if using use_pretrained")
        model = trained
        merged_slurm = {**model.slurm, **slurm}
        model = ModelRef(model.path, model.flag, slurm=merged_slurm)
    else:
        raise ValueError(f"Unknown model type {config['TYPE']}")

    return model


def apply_transition(model, transition, tokenizer, random_seed):
    t_name = transition_path_name(transition)
    tok_id = get_tokenizer_id(tokenizer) if tokenizer is not None else "notok"
    base_dir = f"{model.abspath}-transition/{t_name}"

    if transition["TYPE"] == "none":
        model_dir = f"{base_dir}/{tok_id}/model"
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        flag_file = f"{model_dir}.none_transition.flag"
        node = env.Command(
            target=flag_file,
            source=model.as_source(),
            action="ln -sfn '{}' '{}' && touch $TARGET".format(
                model.abspath, model_dir
            ),
        )
        return ModelRef(Dir(model_dir), flag=node, slurm=model.slurm)
    elif transition["TYPE"] == "reset_embeddings":
        tokenizer_flag = ""
        tokenizer_source = []
        if tokenizer is not None:
            tokenizer_flag = f"--tokenizer {tokenizer[0].abspath}"
            tokenizer_source = [tokenizer]
        std = transition["CONFIG"].get("std", 0.02)
        model_dir = f"{base_dir}/{tok_id}/model"
        flag_file = f"{model_dir}/reset_embeddings.flag"
        node = env.ResetEmbeddings(
            source=model.as_source() + tokenizer_source,
            target=flag_file,
            INPUT_MODEL=model.abspath,
            OUTPUT_DIR=model_dir,
            TOKENIZER_FLAG=tokenizer_flag,
            STD=std,
            **cpu_task_config(env, "reset_emb", "00:30:00", "24GB"),
        )
        return ModelRef(Dir(model_dir), flag=node, slurm=model.slurm)
    else:
        raise ValueError(f"Unknown transition type: {transition['TYPE']}")


# Training

def get_training(model, training_config, scheduler_config, dataset_config, dataset, tokenizer, random_seed, current_stage):
    assert isinstance(model, ModelRef), f"Expected ModelRef, got {type(model)}"

    if dataset_config["TYPE"] in ("pos_tag", "pos_transition"):
        seq_length = training_config["CONFIG"]["SEQUENCE_LENGTH"]
        ds_name_local = dataset_path_name(dataset_config)[0]
        tok_id_local = get_tokenizer_id(tokenizer)
        ds_type = dataset_config["TYPE"]
        c = dataset_config["CONFIG"]
        spacy_model = c.get("SPACY_MODEL", "en_core_web_sm")
        tag_source = c.get("TAG_SOURCE", "upos")
        pos_train_split = dataset_config.get("TRAIN_SPLIT", "train")
        pos_dev_split = dataset_config.get("DEV_SPLIT", "dev")
        if ds_type == "pos_transition" and not c.get("ALIGNED", False):
            raise ValueError(
                "pos_transition requires aligned=True: the per-word pretokenizer was "
                "removed 2026-09-21, and only the aligned builder gives a word side "
                "that is the natural tokenization.")
        tokenized = {"dataset": dataset["dataset"]}

        preprocessed = {}
        if ds_type == "pos_tag":
            corpus_segment = pos_tokenizer_dataset_name(dataset_config)
            chunks_per_split = {"train": 16, "trainA": 16, "trainB": 16, "decay": 2, "dev": 2, "test": 2}
            for key in dataset:
                if key in ("dataset", "dataset_name"):
                    continue
                preprocessed[key] = preprocess_language_split(
                    corpus_segment, random_seed, dataset[key], key,
                    spacy_model=spacy_model, num_chunks=chunks_per_split.get(key, 4))

        for key in dataset:
            if key in ("dataset", "dataset_name"):
                continue
            if ds_type == "pos_tag":
                node = env.PosTagTokenize(
                    source=[preprocessed[key], tokenizer],
                    target=[f"work/datasets/{ds_name_local}/seed{random_seed}/tokenized/{tok_id_local}/seq{seq_length}/{key}.tok"],
                    INPUT=preprocessed[key],
                    TOKENIZER=tokenizer,
                    SEQ_LENGTH=seq_length,
                    TAG_SOURCE=tag_source,
                    RANDOM_SEED=random_seed,
                    **cpu_task_config(env, f"pos_tag_tok_{key}", "12:00:00", "64GB"),
                )
                tokenized[key] = node[0]
                continue

            if dataset_config.get("TRAIN_PREFIX") and key not in (pos_train_split, pos_dev_split):
                continue
            cap = 0
            if key == pos_train_split:
                cap = dataset_config.get("TRAIN_PREFIX") or 0
            elif key == pos_dev_split:
                cap = dataset_config.get("DEV_PREFIX") or 0
            pretok_prefix = (f"work/datasets/{pos_transition_pretok_name(dataset_config)}/"
                             f"seed{random_seed}/pretokenized/{tok_id_local}/{key}")
            node = env.PosPretokenizeAligned(
                source=[dataset[key], tokenizer],
                target=[f"{pretok_prefix}.{k}.bin" for k in POS_STREAM_KEYS],
                INPUT=dataset[key],
                TOKENIZER=tokenizer,
                TAG_SOURCE=tag_source,
                SPACY_MODEL=spacy_model,
                MAX_WORDS_FLAG=(f"--max_words {cap}" if cap else ""),
                **cpu_task_config(env, f"pos_pretok_aligned_{key}", "24:00:00", "32GB"),
            )
            for k, stream_node in zip(POS_STREAM_KEYS, node):
                tokenized[f"{key}_{k}"] = stream_node
        dataset = tokenized

    # Slurm limits are read here and dropped from the config written to
    # training_config.json (an SCons source), so changing them never retrains.
    slurm_time = training_config["CONFIG"].get("SLURM_TIME", "72:00:00")
    slurm_memory = training_config["CONFIG"].get("SLURM_MEMORY", "48GB")
    training_config = copy.deepcopy(training_config)
    training_config["CONFIG"].pop("SLURM_TIME", None)
    training_config["CONFIG"].pop("SLURM_MEMORY", None)

    if dataset_config["TYPE"] == "none":
        output_model_dir = Dir(f"{model.abspath}-transition/none/{current_stage}")
        os.makedirs(output_model_dir.abspath, exist_ok=True)
        copied_model = env.Command(
            target=f"{output_model_dir.abspath}/original_model",
            source=model.as_source(),
            action=Copy("$TARGET", "$SOURCE"),
        )
        training_flag = env.Command(
            target=f"{output_model_dir.abspath}/training_finished.flag",
            source=copied_model,
            action=Touch("$TARGET"),
        )
        return [ModelRef(copied_model, flag=training_flag, slurm=model.slurm)]

    ds_name = dataset_path_name(dataset_config)[0]
    train_tokens = training_config['CONFIG'].get('TRAIN_TOKENS_REQUESTED',
                                                 training_config['CONFIG']['TRAIN_TOKENS'])
    batch_size = training_config['CONFIG']['BATCH_SIZE']
    learning_rate = training_config['HUGGINGFACE_CONFIG']['learning_rate']
    weight_decay = training_config['HUGGINGFACE_CONFIG']['weight_decay']
    lr_str = format_lr(learning_rate)
    unfrozen_top = training_config['CONFIG'].get('UNFROZEN_TOP', 0)
    unfrozen_bottom = training_config['CONFIG'].get('UNFROZEN_BOTTOM', 0)
    if training_config['CONFIG'].get('FREEZE_LAYERS', False):
        freeze_suffix = f"/frozen_t{unfrozen_top}_b{unfrozen_bottom}"
        if training_config['CONFIG'].get('FREEZE_FINAL_NORM', False):
            freeze_suffix += "_normfroz"
    else:
        freeze_suffix = "/unfrozen"
    # /ws only with model_load: legacy path compatibility.
    hf_sched = scheduler_config.get('HUGGINGFACE_CONFIG', {})
    warmup_ratio = hf_sched.get('warmup_ratio')
    warmup_steps_val = hf_sched.get('warmup_steps')
    model_load = training_config['CONFIG'].get('MODEL_LOAD')
    if warmup_ratio is not None and warmup_ratio:
        warmup_suffix = f"/wr{warmup_ratio}"
    elif warmup_steps_val and model_load:
        warmup_suffix = f"/ws{warmup_steps_val}"
    else:
        warmup_suffix = ""
    if model_load:
        _DTYPE_SHORT = {"bfloat16": "bf16", "bf16": "bf16",
                        "float16": "fp16", "fp16": "fp16",
                        "float32": "fp32", "fp32": "fp32"}
        _ATTN_SHORT = {"flash_attention_2": "fa2", "eager": "eager", "sdpa": "sdpa"}
        parts = []
        for k in sorted(model_load.keys()):
            v = model_load[k]
            if k == "torch_dtype":
                parts.append(_DTYPE_SHORT.get(str(v).lower(), str(v)))
            elif k == "attn_implementation":
                parts.append(_ATTN_SHORT.get(str(v), str(v)))
            else:
                parts.append(f"{k}={v}")
        model_load_suffix = "/" + "_".join(parts)
    else:
        model_load_suffix = ""
    transition_schedule = training_config['CONFIG'].get('TRANSITION_SCHEDULE')
    if transition_schedule and (
        'transition_words' in transition_schedule or 'warmup_words' in transition_schedule
    ):
        tw = int(transition_schedule.get('transition_words', 0))
        ww = int(transition_schedule.get('warmup_words', 0))
        tw_part = f"/tw{tw // 1_000_000}M" if tw else ""
        ww_part = f"_ww{ww // 1_000_000}M" if ww else ""
        sp = transition_schedule.get('start_prob')
        ep = transition_schedule.get('end_prob')
        p_part = ""
        if sp is not None and ep is not None and sp == ep and float(sp) != 1.0:
            p_part = f"/pword{sp}"
        transition_suffix = f"{tw_part}{ww_part}{p_part}"
    else:
        transition_suffix = ""
    scale_suffix = ""
    if training_config['CONFIG'].get('SCALE_GRID'):
        _u = int(training_config['CONFIG'].get('TRAIN_TOKENS', 0))
        _ep = int(training_config['HUGGINGFACE_CONFIG'].get('num_train_epochs', 1))
        scale_suffix = f"/u{_u // 1000}K_x{_ep}"
    if training_config["CONFIG"].get("TRAIN_ALL"):
        _prefix_w = dataset_config.get("TRAIN_PREFIX")
        assert _prefix_w, "TRAIN_ALL training requires dataset_config(train_prefix=...) for path naming"
        budget_segment = (f"w{_prefix_w // 1_000_000}M"
                          if _prefix_w % 1_000_000 == 0 else f"w{_prefix_w}")
    else:
        budget_segment = f"tok{format_tokens(train_tokens)}"
    plateau_suffix = ("/plateau"
                      if training_config["CONFIG"].get("TRAIN_ALL")
                      and scheduler_config.get("TYPE") == "reduce_lr_on_plateau"
                      else "")
    _n_epochs = int(training_config["HUGGINGFACE_CONFIG"].get("num_train_epochs", 1))
    epoch_suffix = f"/ep{_n_epochs}" if _n_epochs > 1 else ""
    logeval_suffix = "/logeval" if training_config["CONFIG"].get("LOG_EVAL") else ""
    probe_suffix = "/probe" if training_config["CONFIG"].get("CKPT_PROBE") else ""
    rescale_suffix = (f"/rescale{training_config['CONFIG']['RESCALE_STD']}"
                      if training_config["CONFIG"].get("RESCALE_STD") is not None else "")
    readable_prefix = f"{ds_name}/{budget_segment}/bs{batch_size}/lr{lr_str}/wd{weight_decay}{warmup_suffix}{epoch_suffix}{plateau_suffix}{logeval_suffix}{probe_suffix}{rescale_suffix}{model_load_suffix}{freeze_suffix}{transition_suffix}{scale_suffix}/seed{random_seed}"

    output_model_dir = Dir(f"{os.path.dirname(model.abspath)}/{readable_prefix}/{current_stage}")

    os.makedirs(output_model_dir.abspath, exist_ok=True)

    train_split = dataset_config.get("TRAIN_SPLIT", "train")
    dev_split = dataset_config.get("DEV_SPLIT", "dev")
    if dataset_config["TYPE"] == "pos_transition":
        train_data = dataset[f"{train_split}_word_ids"]
        dev_data = dataset[f"{dev_split}_word_ids"]
        data_sources = [dataset[f"{split}_{k}"]
                        for split in (train_split, dev_split) for k in POS_STREAM_KEYS]
    else:
        train_data = dataset[train_split]
        dev_data = dataset[dev_split]
        data_sources = [train_data, dev_data]
    config = {
        "DATASET_CONFIG": dataset_config,
        "SCHEDULER_CONFIG": scheduler_config,
        "TRAINING_CONFIG": training_config,
        "TOKENIZER": tokenizer[0].abspath,
        "random_seed": random_seed,
        "train_dataset": train_data.abspath,
        "dev_dataset": dev_data.abspath,
    }

    update_config(f"{output_model_dir.abspath}/scons_config.json", config)

    config_file = f"{output_model_dir}/training_config.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    update_config(config_file, training_config)
    scheduler_file = f"{output_model_dir}/scheduler_config.json"
    update_config(scheduler_file, scheduler_config)
    dataset_file = f"{output_model_dir}/dataset_config.json"
    update_config(dataset_file, dataset_config)

    all_checkpoints = [Dir(f"{output_model_dir.abspath}/{name}")
                       for name in checkpoint_dir_names(training_config, dataset_config)]

    training_outputs = env.TrainModel(
        source=model.as_source() + data_sources + [tokenizer, config_file, scheduler_file, dataset_file],
        target=[f"{output_model_dir}/training_finished.flag"],
        MODEL=model.abspath,
        OUTPUT_DIR=output_model_dir,
        TRAIN_DATA=train_data,
        DEV_DATA=dev_data,
        TRAINING_CONFIG=config_file,
        SCHEDULER_CONFIG=scheduler_file,
        DATASET_CONFIG=dataset_file,
        TOKENIZER=tokenizer,
        RANDOM_SEED=random_seed,
        **{**gpu_task_config(env, "train_model", slurm_time, slurm_memory), **model.slurm},
    )

    if training_config["CONFIG"].get("RETURN_CHECKPOINTS", False):
        trained_model = all_checkpoints[-1]
    else:
        trained_model = all_checkpoints

    trained_model = [ModelRef(model, training_outputs) for model in trained_model]
    return trained_model


# Evaluation

def get_evaluation(eval_configs, trained_models, training_config, dataset_config, tokenizer, random_seed):
    if not eval_configs or not trained_models:
        return []

    training_dir = os.path.dirname(trained_models[0].abspath)
    all_checkpoint_models = {os.path.join(training_dir, name): None
                             for name in checkpoint_dir_names(training_config, dataset_config)}

    for model_ref in trained_models:
        if model_ref.abspath in all_checkpoint_models:
            all_checkpoint_models[model_ref.abspath] = model_ref

    shared_flag = trained_models[0].flag

    all_eval_outputs = []
    for eval_cfg in eval_configs:
        eval_type = eval_cfg["TYPE"]
        run_all = eval_cfg["CONFIG"].get("ALL_CHECKPOINTS", False)
        eval_slurm = eval_cfg["CONFIG"].get("SLURM", {})

        eval_split = eval_cfg["CONFIG"].get("split", None)
        eval_data_path = eval_cfg["CONFIG"].get("data_path", None)
        eval_tasks = eval_cfg["CONFIG"].get("tasks", None)
        suffix_parts = []
        if eval_data_path:
            suffix_parts.append(os.path.splitext(os.path.basename(eval_data_path))[0])
        if eval_split:
            suffix_parts.append(eval_split)
        if eval_tasks:
            suffix_parts.append(eval_tasks if isinstance(eval_tasks, str) else "_".join(eval_tasks))
        suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
        eval_name = f"{eval_type}{suffix}"

        if eval_type not in EVAL_TYPES:
            raise ValueError(f"Unknown eval type '{eval_type}'. Register it in EVAL_TYPES.")
        builder_name, builder_vars = EVAL_TYPES[eval_type]
        builder_fn = getattr(env, builder_name)

        eval_config_path = f"{training_dir}/eval_{eval_name}_config.json"
        os.makedirs(training_dir, exist_ok=True)
        eval_config_file = update_config(eval_config_path, eval_cfg)

        if run_all:
            checkpoint_dirs = list(all_checkpoint_models.keys())
        else:
            checkpoint_dirs = [list(all_checkpoint_models.keys())[-1]] if all_checkpoint_models else []

        for cp_dir in checkpoint_dirs:
            model_ref = all_checkpoint_models.get(cp_dir)
            source_dep = model_ref.as_source() if model_ref else [shared_flag]

            output_path = f"{cp_dir}/eval_{eval_name}.json"
            eval_output = builder_fn(
                source=source_dep + [tokenizer, eval_config_file],
                target=[output_path],
                CHECKPOINT=cp_dir,
                TOKENIZER=tokenizer,
                EVAL_CONFIG=eval_config_path,
                **builder_vars,
                # Walltime is in the action signature; changing it re-runs evals.
                **{**gpu_task_config(env, f"eval_{eval_type}", "06:00:00", "48GB"), **eval_slurm},
            )
            all_eval_outputs.append(eval_output)

    return all_eval_outputs


# Main loop

for random_seed in env["RANDOM_SEEDS"]:
    # Never set env["RANDOM_SEED"]: substituted at execution with the last value.
    for experiment_name, experiment in env["EXPERIMENTS"].items():
        past_models = [None]
        stages = {}
        tokenizer = None
        tok_name = None
        tokenized_dataset = None

        for stage in list(sorted(list(experiment.keys()))):
            stages[stage] = None
            print(f"Processing stage {stage} for experiment {experiment_name} with random seed {random_seed}")
            combinations = list(get_combinations(experiment[stage], parents=past_models))
            past_models = []
            for combination in combinations:
                parent_model = None
                parent_tokenizer = None
                parent_tok_name = None
                if "parent" in combination:
                    parent_model = combination["parent"][0]
                    parent_tokenizer, parent_tok_name = combination["parent"][1]
                model = get_model(combination["model"], random_seed, parent_model)
                if not combination["dataset"]["TYPE"] == "none":
                    ds_name = dataset_path_name(combination["dataset"])[0]
                    dataset = get_dataset(combination["dataset"], random_seed)
                    effective_ds_name = dataset.get("dataset_name", ds_name)
                    is_scramble = combination["dataset"]["TYPE"] in ("pos_tag", "pos_transition")
                    if is_scramble:
                        tok_ds_name = pos_tokenizer_dataset_name(combination["dataset"])
                    else:
                        tok_ds_name = effective_ds_name
                    tokenizer, tok_name = get_tokenizer(
                        combination["model"], dataset=dataset["dataset"], dataset_name=tok_ds_name, pretrained_tokenizer=parent_tokenizer, pretrained_tok_name=parent_tok_name, random_seed=random_seed, dataset_splits=dataset, dataset_config=combination["dataset"])
                    if is_scramble:
                        tokenized_dataset = dataset
                    else:
                        tokenized_dataset = get_tokenized_datasets(dataset, tokenizer, effective_ds_name, tok_name, random_seed, dataset_config=combination["dataset"])
                    transition = combination.get("transition", {"TYPE": "none", "CONFIG": {}})
                    model= apply_transition(model, transition, tokenizer, random_seed)
                effective_training = combination["training"]
                training_overrides = combination["model"]["CONFIG"].get("TRAINING_OVERRIDES", {})
                if training_overrides:
                    effective_training = copy.deepcopy(combination["training"])
                    for key, value in training_overrides.items():
                        if key == "gradient_accumulation_steps":
                            batch_size = effective_training["CONFIG"]["BATCH_SIZE"]
                            gpus = effective_training["CONFIG"]["GPUS"]
                            device_num = len(gpus.split(","))
                            batch_size_per_device = batch_size // device_num // value
                            assert batch_size_per_device * value * device_num == batch_size, \
                                f"Batch size not divisible by devices and gradient_accumulation_steps: {batch_size} {device_num} {value}"
                            effective_training["HUGGINGFACE_CONFIG"]["per_device_train_batch_size"] = int(batch_size_per_device)
                            effective_training["HUGGINGFACE_CONFIG"]["per_device_eval_batch_size"] = int(batch_size_per_device)
                        effective_training["HUGGINGFACE_CONFIG"][key] = value

                pre_pre_trained_model = get_training(
                    model,
                    effective_training,
                    combination["scheduler"],
                    combination["dataset"],
                    tokenized_dataset,
                    tokenizer,
                    random_seed=random_seed,
                    current_stage=stage,
                )
                model_tokenizer_pairs = [(model, (tokenizer, tok_name)) for model in pre_pre_trained_model]
                past_models.extend(model_tokenizer_pairs)

                get_evaluation(
                    combination.get("evaluation", []),
                    pre_pre_trained_model,
                    effective_training,
                    combination["dataset"],
                    tokenizer,
                    random_seed,
                )
            stages[stage] = past_models
