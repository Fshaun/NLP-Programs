"""
Text Preprocessing Pipeline for a Clinical Text Classifier
============================================================

Chains raw_text -> custom_tokenize -> remove_stopwords -> lemmatize_tokens
and prints the intermediate state after every stage.

Only nltk (for POS-aware lemmatisation) is used as an external dependency,
as explicitly permitted by section 4 of the assignment. Tokenisation and
stopword filtering use only native Python / re, per the stated constraint.
"""

import re
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

raw_text = (
    "The intensive-care patients are recovering surprisingly quickly. "
    "Doctors are analyzing data daily!"
)


# ---------------------------------------------------------------------------
# 2. Custom Tokenisation and Case Normalisation
# ---------------------------------------------------------------------------
def custom_tokenize(text):
    """
    Convert `text` to a clean list of lowercased tokens using only native
    string methods / the `re` module (no NLTK/SpaCy/HuggingFace tokenizers).

    Linguistic design choice:
    Hyphenated clinical terms (e.g. "intensive-care") are kept BOUND as a
    single token rather than split into "intensive" and "care". In clinical
    text, hyphenated compounds are frequently a single semantic unit (a
    modifier + noun acting as one concept), and splitting them would lose
    that meaning for a downstream classifier (e.g. "intensive" alone is a
    generic adjective, but "intensive-care" names a specific ward/context).
    We therefore treat the hyphen as part of the word character class
    instead of as a delimiter.
    """
    text = text.lower()

    # \w matches [a-z0-9_]; we add hyphen so hyphenated compounds are not
    # broken apart. This regex extracts runs of letters/digits/hyphens,
    # which has the side effect of stripping surrounding punctuation
    # (commas, periods, exclamation marks) automatically, since those
    # characters are not part of the match.
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

    return tokens


# ---------------------------------------------------------------------------
# 3. Deterministic Stopword Filtering
# ---------------------------------------------------------------------------
# Hardcoded stopword set relevant to the target snippet.
CUSTOM_STOPWORDS = {"the", "are", "is", "a", "an", "and", "of", "to", "in"}


def remove_stopwords(token_list, custom_stopwords):
    """
    Strip structural/grammatical words from `token_list` using
    `custom_stopwords`. Implemented as a list comprehension for efficient,
    single-pass filtering (O(n) with O(1) set membership checks).
    """
    return [token for token in token_list if token not in custom_stopwords]


# ---------------------------------------------------------------------------
# 4. Context-Aware Lemmatisation
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()

_PENN_TO_WORDNET = {
    "J": wordnet.ADJ,
    "V": wordnet.VERB,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
}


def _get_wordnet_pos(penn_tag):
    """Map a Penn Treebank POS tag's first letter to a WordNet POS tag."""
    return _PENN_TO_WORDNET.get(penn_tag[0].upper(), wordnet.NOUN)


def lemmatize_tokens(token_list):
    """
    Lemmatise each token using WordNetLemmatizer, but crucially compute the
    correct POS tag for every token first via nltk.pos_tag.

    A naive lemmatiser defaults to noun mode, so "recovering" and
    "analyzing" (both verbs, gerund form -ing) would be returned unchanged.
    By tagging first and passing pos=wordnet.VERB where appropriate, verbs
    are correctly reduced to their base form (e.g. "recovering" -> "recover").
    """
    tagged_tokens = pos_tag(token_list)  # [(token, penn_tag), ...]

    lemmas = [
        _lemmatizer.lemmatize(token, pos=_get_wordnet_pos(tag))
        for token, tag in tagged_tokens
    ]
    return lemmas


# ---------------------------------------------------------------------------
# 5. Pipeline Orchestration and State Verification
# ---------------------------------------------------------------------------
def run_pipeline(text):
    print("Stage 0 - raw_text:")
    print(f"  {text!r}\n")

    tokens = custom_tokenize(text)
    print("Stage 1 - after custom_tokenize:")
    print(f"  {tokens}\n")

    filtered_tokens = remove_stopwords(tokens, CUSTOM_STOPWORDS)
    print("Stage 2 - after remove_stopwords:")
    print(f"  {filtered_tokens}\n")

    lemmas = lemmatize_tokens(filtered_tokens)
    print("Stage 3 - after lemmatize_tokens:")
    print(f"  {lemmas}\n")

    return lemmas


if __name__ == "__main__":
    run_pipeline(raw_text)