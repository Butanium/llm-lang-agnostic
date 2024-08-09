from __future__ import annotations

import pandas as pd
from warnings import warn
from cache_decorator import Cache
import ast
from emoji import emoji_count

import babelnet as bn
from babelnet.sense import BabelLemmaType, BabelSense
from babelnet import BabelSynset


from babelnet.api import BabelAPIType, _api_type
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def BabelCache(**kwargs):
    def deco(func):
        cached_func = Cache(**kwargs)(func)

        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            if result == []:
                res_type = list
            else:
                res_type = type(result[0]) if isinstance(result, list) else type(result)
            if "Online" in str(res_type) and _api_type == BabelAPIType.RPC:
                # If we are using the RPC API, we can delete the cache and save the offline data instead
                path = Path(Cache.compute_path(cached_func, *args, **kwargs))
                path.unlink()
            return cached_func(*args, **kwargs)

        return wrapper

    return deco


id_to_bn_lang = {
    "en": bn.Language("English"),
    "es": bn.Language("Spanish"),
    "fr": bn.Language("French"),
    "de": bn.Language("German"),
    "it": bn.Language("Italian"),
    "ja": bn.Language("Japanese"),
    "ru": bn.Language("Russian"),
    "zh": bn.Language("Chinese"),
    "ko": bn.Language("Korean"),
    "nl": bn.Language("Dutch"),
    "et": bn.Language("Estonian"),
    "fi": bn.Language("Finnish"),
    "hi": bn.Language("Hindi"),
}
lang_to_id = {v: k for k, v in id_to_bn_lang.items()}


@BabelCache()
def cached_synset_from_id(synset_id, to_langs=None):
    id = bn.BabelSynsetID(synset_id)
    return bn.get_synset(
        id, to_langs=to_langs and [id_to_bn_lang[to_lang] for to_lang in to_langs]
    )


@BabelCache()
def cached_bn_synsets(word, from_langs, poses=None, to_langs=None):
    return bn.get_synsets(
        word,
        from_langs=[id_to_bn_lang[from_lang] for from_lang in from_langs],
        to_langs=[id_to_bn_lang[to_lang] for to_lang in to_langs],
        poses=poses,
    )


@BabelCache(
    args_to_ignore=("sense_filters", "synset_filters"),
)
def cached_bn_senses(
    word, from_langs, to_langs, sense_filters=None, synset_filters=None, poses=None
):
    return bn.get_senses(
        word,
        sense_filters=sense_filters,
        synset_filters=synset_filters,
        from_langs=[id_to_bn_lang[from_lang] for from_lang in from_langs],
        to_langs=[id_to_bn_lang[to_lang] for to_lang in to_langs],
        poses=poses,
    )


def get_synset_from_id(synset_id, to_langs=None):
    if isinstance(to_langs, str):
        to_langs = [to_langs]
    if to_langs is not None:
        to_langs = sorted(to_langs)
    return cached_synset_from_id(synset_id, to_langs=to_langs)


def get_bn_synsets(word, from_langs, poses=None, to_langs=None):
    if isinstance(from_langs, str):
        from_langs = [from_langs]
    from_langs = sorted(from_langs)
    if to_langs is None:
        to_langs = from_langs
    else:
        to_langs = sorted(to_langs)
    return cached_bn_synsets(word, from_langs, poses=poses, to_langs=to_langs)


def get_bn_senses(
    word, from_langs, to_langs, sense_filters=None, synset_filters=None, poses=None
):
    if isinstance(from_langs, str):
        from_langs = [from_langs]
    if isinstance(to_langs, str):
        to_langs = [to_langs]
    from_langs = sorted(from_langs)
    to_langs = sorted(to_langs)
    return cached_bn_senses(
        word,
        from_langs,
        to_langs,
        sense_filters=sense_filters,
        synset_filters=synset_filters,
        poses=poses,
    )


def _synset_filter(synset: BabelSynset, key_concept_only=False):
    if synset.type != bn.synset.SynsetType.CONCEPT:
        return False  # Removes albums, movies, etc.
    if key_concept_only and not synset.is_key_concept:
        return False
    return True


def filter_synsets(synsets, key_concept_only=False):
    return [synset for synset in synsets if _synset_filter(synset, key_concept_only)]


def _sense_filter(sense: BabelSense):
    lemma = sense.lemma.lemma
    if (
        sense._lemma.lemma_type != BabelLemmaType.HIGH_QUALITY
        or emoji_count(lemma) > 0
        or any(char.isdigit() for char in lemma)  # Removes emojis
    ):  # Remove numbers
        return False
    return True


def filter_senses(senses):
    return list(filter(_sense_filter, senses))


def bn_translate(
    word,
    source_lang,
    target_langs,
    noun_only=True,
    key_concept_only=True,
    keep_original_word=False,
):
    def sense_filter(sense: BabelSense):
        lemma = sense.lemma.lemma
        if (
            sense._lemma.lemma_type
            != BabelLemmaType.HIGH_QUALITY  # Removes low-quality translations
            or (
                not keep_original_word and lemma.lower() == word.lower()
            )  # Removes the original word
            or emoji_count(lemma) > 0  # Removes emojis
            or any(char.isdigit() for char in lemma)  # Remove numbers
        ):
            return False  # Removes low-quality translations
        return True

    if isinstance(target_langs, str):
        target_langs = [target_langs]

    kwargs = dict(
        from_langs=[source_lang],
        to_langs=target_langs,
    )
    if noun_only:
        kwargs["poses"] = [bn.POS.NOUN]
    senses = get_bn_senses(word, **kwargs)
    senses = [sense for sense in senses if sense_filter(sense)]
    qualified_senses = [s for s in senses if _synset_filter(s.synset)]
    max_degree = max(
        [sense.synset.synset_degree for sense in qualified_senses], default=0
    )
    best_senses = [
        sense for sense in qualified_senses if sense.synset.synset_degree == max_degree
    ]
    if key_concept_only:
        qualified_senses = [
            sense for sense in senses if _synset_filter(sense.synset, key_concept_only)
        ]
    if qualified_senses == [] and key_concept_only:
        warn(f"Didn't find any key concept for {word}, adding the best non-key concept")
    qualified_senses += best_senses
    translations = {lang: [] for lang in target_langs}
    for sense in qualified_senses:
        translations[lang_to_id[sense.language]].append(sense)
    for lang in target_langs:
        # Sorting like this puts the most common translations first :
        # "In practice, this connectivity measure weights a sense as more appropriate if it has a high degree"
        sort = sorted(
            translations[lang],
            key=lambda s: s.synset.synset_degree,
            reverse=True,
        )
        translations[lang] = list(
            dict.fromkeys([sense.lemma.lemma.replace("_", " ") for sense in sort])
        )
    return translations


def filter_translations(
    translation_df, tok_vocab=None, multi_token_only=False, single_token_only=False
):
    assert not (
        multi_token_only and single_token_only
    ), "Cannot have both multi_token_only and single_token_only"
    assert tok_vocab is not None or not (
        multi_token_only or single_token_only
    ), "Cannot filter tokens without a tokenizer"
    if not multi_token_only and not single_token_only:
        return translation_df
    for idx, row in translation_df.iterrows():
        for lang in translation_df.columns:
            if row[lang] in tok_vocab or "▁" + row[lang] in tok_vocab:
                if multi_token_only:
                    translation_df.drop(idx, inplace=True)
                    break
            elif single_token_only:
                translation_df.drop(idx, inplace=True)
                break
    print(f"Filtered to {len(translation_df)} translations")
    return translation_df
