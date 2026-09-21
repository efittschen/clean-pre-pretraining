"""Run-path naming for the build graph.  Hyperparameters live in the path;
renaming one orphans every existing result."""
import itertools
import os

# The four files of an aligned split, in write order.
POS_STREAM_KEYS = ("word_ids", "word_offsets", "pos_ids", "pos_offsets")


def dataset_path_name(config, all=False):
    t = config["TYPE"]
    c = config["CONFIG"]
    words_M = config["WORDS"] // 1_000_000
    if t == "language":
        return [f"language/{c['LANGUAGE']}/{words_M}M"]
    elif t == "c4":
        return [f"c4/{c['C4_CONFIG']}/{words_M}M"]
    elif t == "babylm":
        # Train and dev repos differ, so both are encoded.
        repo_tag = c["BABYLM_REPO"].split("/")[-1]
        wcap = f"{words_M}M" if c.get("WORDS") else "full"
        dev_repo = c.get("BABYLM_DEV_REPO", c["BABYLM_REPO"])
        dev_segment = ""
        if dev_repo != c["BABYLM_REPO"]:
            dev_segment = f"/dev_{dev_repo.split('/')[-1]}"
        return [f"babylm/{repo_tag}/{c['BABYLM_CONFIG'] or 'default'}/{wcap}{dev_segment}"]
    elif t == "shuffle_dyck":
        return [f"shuffle_dyck/sym{c['NUM_SYMBOLS']}/len{c['TARGET_LENGTH']}/p{c['P']}/{words_M}M"]
    elif t == "dyck":
        return [f"dyck/sym{c['NUM_SYMBOLS']}/len{c['TARGET_LENGTH']}/d{c['MIN_DEPTH']}-{c['MAX_DEPTH']}/{words_M}M"]
    elif t in ("wikitext"):
        return [t]
    elif t == "text_file":
        return [c["FILE_NAME"]]
    elif t == "pos_tag":
        return [f"pos_tag/{pos_corpus_segment(config)}/{c.get('TAG_SOURCE', 'upos')}"]
    elif t == "pos_tag_text":
        source_segment = dataset_path_name(c["SOURCE"])[0]
        return [f"pos_tag_text/{source_segment}/{c.get('TAG_SOURCE', 'ptb')}"]
    elif t == "pos_transition":
        wpd = c.get('WORDS_PER_DATAPOINT', 0)
        return [f"pos_transition/{pos_corpus_segment(config)}/{pos_transition_tag_segment(config)}_wpd{wpd}"]
    elif t == "ngram":
        n = c["N"]
        src_lang = c["SOURCE_LANGUAGE"]
        src_words_M = c["SOURCE_WORDS"] // 1_000_000
        vocab_cap = c.get("VOCAB_CAP", 100_000)
        return [f"ngram/n{n}/from_{src_lang}_{src_words_M}M_cap{vocab_cap}/{words_M}M"]
    elif t == "none":
        return ["none"]
    else:
        if not all:
            return [t]
        elif t in ("pos", "pos_simple", "pos_text"):
            return ("pos", "pos_simple", "pos_text")
        elif t in ("bnc_text", "bnc_tags"):
            return ("bnc_text", "bnc_tags")
    return t


# Source mode (any corpus) vs legacy language mode: decided here only.

def pos_corpus_segment(config):
    c = config["CONFIG"]
    source = c.get("SOURCE")
    if source is not None:
        return dataset_path_name(source)[0]
    return f"{c['LANGUAGE']}/{config['WORDS'] // 1_000_000}M"


def pos_tokenizer_dataset_name(config):
    c = config["CONFIG"]
    source = c.get("SOURCE")
    if source is not None:
        return dataset_path_name(source)[0]
    return f"language/{c['LANGUAGE']}/{config['WORDS'] // 1_000_000}M"


def pos_transition_tag_segment(config):
    c = config["CONFIG"]
    tag = c.get("TAG_SOURCE", "upos")
    return f"{tag}_aligned" if c.get("ALIGNED", False) else tag


def pos_transition_pretok_name(config):
    seg = pos_transition_tag_segment(config)
    dev_prefix = config.get("DEV_PREFIX")
    if config["CONFIG"].get("ALIGNED", False) and dev_prefix:
        seg += f"_dev{dev_prefix // 1_000_000}Mw"
    return f"pos_transition/{pos_corpus_segment(config)}/{seg}"


def get_tokenizer_id(tokenizer):
    tok_path = tokenizer[0].abspath
    tok_marker = "/tokenizers/"
    idx = tok_path.find(tok_marker)
    if idx != -1:
        return tok_path[idx + len(tok_marker):].replace("/", "_")
    return os.path.basename(tok_path)


def format_lr(learning_rate):
    if learning_rate >= 1e-2:
        return f"{learning_rate:.3f}".rstrip('0').rstrip('.')
    return f"{learning_rate:.0e}"


def get_combinations(stages, parents=[None], parent_key="parent"):
    non_product_keys = {"evaluation"}
    for stage in stages:
        keys = [k for k in stage.keys() if k not in non_product_keys]
        extra = {k: stage[k] for k in stage if k in non_product_keys}
        for values in itertools.product(*(stage[k] for k in keys)):
            base = dict(zip(keys, values))
            base.update(extra)
            if parents == [None]:
                yield base
            else:
                for p in parents:
                    combo = dict(base)
                    combo[parent_key] = p
                    yield combo


def transition_path_name(transition):
    return transition["TYPE"]
