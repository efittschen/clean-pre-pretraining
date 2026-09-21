"""POS tag inventories, shared by the dataset builders, the tokenizer
extensions and the masked perplexity evaluator."""

PTB_NON_PUNCT_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB",
]
PTB_PUNCT_TAGS = [
    ".", ",", ":", ";", "?", "!", "''", "``",
    "-LRB-", "-RRB-", "(", ")", '"', "HYPH", "NFP",
]

# Only PUNCT is no-leading-space.
UPOS_NON_PUNCT_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X", "SPACE",
]
UPOS_PUNCT_TAGS = ["PUNCT"]

# Private-use prefix so tag tokens never match natural text.
POS_TAG_MARKER = "\ue000"


def pos_tag_extras(non_punct_tags, punct_tags, marker=""):
    extras = []
    for tag in non_punct_tags:
        extras.append(marker + tag)
        extras.append(marker + " " + tag)
    for tag in punct_tags:
        extras.append(marker + tag)
    return extras


def pos_tag_extras_for(tag_source, marker=False):
    m = POS_TAG_MARKER if marker else ""
    if tag_source == "ptb":
        return pos_tag_extras(PTB_NON_PUNCT_TAGS, PTB_PUNCT_TAGS, m)
    if tag_source == "upos":
        return pos_tag_extras(UPOS_NON_PUNCT_TAGS, UPOS_PUNCT_TAGS, m)
    raise ValueError(f"Unknown tag_source: {tag_source!r}")


