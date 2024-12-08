import pandas as pd
from random import sample, shuffle
from itertools import product
from copy import deepcopy
from typing import Optional, Callable
from dataclasses import dataclass
from .utils import (
    get_tokenizer,
    ulist,
    process_tokens,
    process_tokens_with_tokenization,
)
import torch as th
import re


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_strings: str
    latent_strings: dict[str, str | list[str]]
    input_string: Optional[str | list[str]] = None

    @classmethod
    def from_strings(
        cls, prompt, target_strings, latent_strings, tokenizer, augment_token=False
    ):
        tok_vocab = tokenizer.get_vocab()
        process_toks = (
            (lambda s: process_tokens(s, tok_vocab))
            if augment_token
            else (lambda s: process_tokens_with_tokenization(s, tokenizer))
        )
        tokenizer = get_tokenizer(tokenizer)
        target_tokens = process_toks(target_strings)
        latent_tokens = {
            lang: process_toks(words) for lang, words in latent_strings.items()
        }
        return cls(
            target_tokens=target_tokens,
            latent_tokens=latent_tokens,
            target_strings=target_strings,
            latent_strings=latent_strings,
            prompt=prompt,
        )

    def get_target_probs(self, probs):
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        return target_probs.cpu()

    def get_latent_probs(self, probs, layer=None):
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2).cpu()
            for lang, tokens in self.latent_tokens.items()
        }
        if layer is not None:
            latent_probs = {
                lang: probs_[:, layer] for lang, probs_ in latent_probs.items()
            }
        return latent_probs

    @th.no_grad
    def run(self, nn_model, get_probs: Callable):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target and latent languages.
        """
        probs = get_probs(nn_model, self.prompt)
        return self.get_target_probs(probs), self.get_latent_probs(probs)

    def has_no_collisions(self, ignore_langs: Optional[str | list[str]] = None):
        tokens = self.target_tokens[:]  # Copy the list
        if isinstance(ignore_langs, str):
            ignore_langs = [ignore_langs]
        if ignore_langs is None:
            ignore_langs = []
        for lang, lang_tokens in self.latent_tokens.items():
            if lang in ignore_langs:
                continue
            tokens += lang_tokens
        return len(tokens) == len(set(tokens))


lang2name = {
    "fr": "Français",
    "de": "Deutsch",
    "ru": "Русский",
    "en": "English",
    "zh": "中文",
    "es": "Español",
    "ko": "한국어",
    "ja": "日本語",
    "it": "Italiano",
    "nl": "Nederlands",
    "et": "Eesti",
    "fi": "Suomi",
    "hi": "हिन्दी",
    "A": "A",
    "B": "B",
}


def get_df_iterrrows(df, words, words_column):
    if words is None:
        iterrrows = df.iterrows()
    else:
        if isinstance(words, str):
            words = [words]
        words_idx = [df.index[df[words_column] == w] for w in words]
        for i, word in enumerate(words):
            if len(words_idx[i]) == 0:
                raise ValueError(f"Word {word} not found in dataframe")
            if len(words_idx[i]) > 1:
                raise ValueError(f"Word {word} found multiple times in dataframe")
        words_idx = [w[0] for w in words_idx]
        rows = [df.loc[idx] for idx in words_idx]
        iterrrows = zip(words_idx, rows)
    return iterrrows


def prompts_from_df(
    input_lang: str,
    target_lang: str,
    df: pd.DataFrame,
    n: int = 5,
    input_lang_name=None,
    target_lang_name=None,
    cut_at_obj=False,
    words: None | str | list[str] = None,
    words_column="word_original",
):
    prompts = []
    pref_input = (
        input_lang_name if input_lang_name is not None else lang2name[input_lang]
    )
    pref_target = (
        target_lang_name if target_lang_name is not None else lang2name[target_lang]
    )
    if pref_input:
        pref_input += ": "
    if pref_target:
        pref_target += ": "
    for idx, row in get_df_iterrrows(df, words, words_column):
        idxs = df.index.tolist()
        idxs.remove(idx)
        fs_idxs = sample(idxs, n)
        prompt = ""
        for fs_idx in fs_idxs:
            fs_row = df.loc[fs_idx]
            in_word = fs_row[input_lang]
            target_word = fs_row[target_lang]
            if isinstance(in_word, list):
                in_word = in_word[0]
            if isinstance(target_word, list):
                target_word = target_word[0]
            prompt += f'{pref_input}"{in_word}" - {pref_target}"{target_word}"\n'
        in_word = row[input_lang]
        if isinstance(in_word, list):
            in_word = in_word[0]
        prompt += f'{pref_input}"{in_word}'
        if not cut_at_obj:
            prompt += f'" - {pref_target}"'
        prompts.append(prompt)
    return prompts


def translation_prompts(
    df,
    tokenizer,
    input_lang: str,
    target_lang: str,
    latent_langs: str | list[str] | None = None,
    n=5,
    only_best=False,
    augment_tokens=False,
    input_lang_name=None,
    target_lang_name=None,
    cut_at_obj=False,
    return_strings=False,
    words: None | str | list[str] = None,
    words_column="word_original",
) -> list[Prompt] | list[str]:
    """
    Get a translation prompt from input_lang to target_lang for each row in the dataframe.

    Args:
        df: DataFrame containing translations
        tokenizer: Huggingface tokenizer
        input_lang: Language to translate from
        target_lang: Language to translate to
        n: Number of few-shot examples for each translation
        only_best: If True, only use the best translation for each row
        augment_tokens: If True, take the subwords, _word for each word

    Returns:
        List of Prompt objects
    """
    tok_vocab = tokenizer.get_vocab()
    if isinstance(latent_langs, str):
        latent_langs = [latent_langs]
    if latent_langs is None:
        latent_langs = []
    assert (
        len(df) > n
    ), f"Not enough translations from {input_lang} to {target_lang} for n={n}"
    prompts = []
    prompts_str = prompts_from_df(
        input_lang,
        target_lang,
        df,
        n=n,
        input_lang_name=input_lang_name,
        target_lang_name=target_lang_name,
        cut_at_obj=cut_at_obj,
        words=words,
        words_column=words_column,
    )
    if return_strings:
        return prompts_str
    for prompt, (_, row) in zip(prompts_str, get_df_iterrrows(df, words, words_column)):
        target_words = row[target_lang]
        if only_best and isinstance(target_words, list):
            target_words = target_words[0]
        if augment_tokens:
            target_tokens = process_tokens(target_words, tok_vocab)
        else:
            target_tokens = process_tokens_with_tokenization(target_words, tokenizer)
        latent_tokens = {}
        latent_words = {}
        for lang in latent_langs:
            l_words = row[lang]
            if only_best and isinstance(l_words, list):
                l_words = l_words[0]
            latent_words[lang] = l_words
            if augment_tokens:
                latent_tokens[lang] = process_tokens(l_words, tok_vocab)
            else:
                latent_tokens[lang] = process_tokens_with_tokenization(
                    l_words, tokenizer
                )
        if len(target_tokens) and all(
            len(latent_tokens_) for latent_tokens_ in latent_tokens.values()
        ):
            prompts.append(
                Prompt(
                    prompt,
                    target_tokens,
                    latent_tokens,
                    target_words,
                    latent_words,
                    row[input_lang],
                )
            )
    return prompts


def random_prompts(df, tokenizer, n=5, **kwargs):
    """
    Given a df with several languages, generate a prompt where there is no logical connection between
    the languages and concepts. E.g.
    A: "apple" - B: "chien"
    A: "柠檬" - B: "Käse"
    """
    langs = [col for col in df.columns if col != "word_original"]
    concepts = []
    for _, row in df.iterrows():
        for lang in langs:
            concepts.append(row[lang])
    concepts2 = concepts.copy()
    shuffle(concepts)
    shuffle(concepts2)
    df2 = pd.DataFrame({"A": concepts, "B": concepts2})
    prompts = translation_prompts(df2, tokenizer, "A", "B", n=n, **kwargs)
    for p in prompts:
        p.target_tokens = []
        p.target_strings = []
    return prompts


def def_prompt(
    df,
    tokenizer,
    lang,
    latent_langs=None,
    use_word_to_def=False,
    words=None,
    words_column="word_original",
    **kwargs,
):
    if latent_langs is None:
        latent_langs = []
    prompts = translation_prompts(
        df,
        tokenizer,
        f"definitions_wo_ref_{lang}" if not use_word_to_def else f"senses_{lang}",
        f"senses_{lang}" if not use_word_to_def else f"definitions_wo_ref_{lang}",
        [
            f"senses_{l}" if not use_word_to_def else f"definitions_wo_ref_{l}"
            for l in latent_langs
        ]
        + [f"senses_{lang}", f"definitions_wo_ref_{lang}"],
        input_lang_name="",
        target_lang_name="",
        words=words,
        words_column=words_column,
        **kwargs,
    )
    for p, (_, row) in zip(prompts, get_df_iterrrows(df, words, words_column)):
        p.target_strings = (
            p.latent_strings[f"senses_{lang}"]
            if not use_word_to_def
            else p.latent_strings[f"definitions_wo_ref_{lang}"]
        )
        p.target_tokens = (
            p.latent_tokens[f"senses_{lang}"]
            if not use_word_to_def
            else p.latent_tokens[f"definitions_wo_ref_{lang}"]
        )
        p.input_string = (
            row[f"senses_{lang}"]
            if use_word_to_def
            else row[f"definitions_wo_ref_{lang}"]
        )
        if isinstance(p.input_string, list):
            p.input_string = p.input_string[0]
        p.latent_tokens = {
            k.split("_")[-1]: v
            for k, v in p.latent_tokens.items()
            if not (
                k == f"senses_{lang}"
                and not use_word_to_def
                and lang not in latent_langs
            )
            and not (
                k == f"definitions_wo_ref_{lang}"
                and use_word_to_def
                and lang not in latent_langs
            )
        }
        p.latent_strings = {
            k.split("_")[-1]: v
            for k, v in p.latent_strings.items()
            if not (
                k == f"senses_{lang}"
                and not use_word_to_def
                and lang not in latent_langs
            )
            and not (
                k == f"definitions_wo_ref_{lang}"
                and use_word_to_def
                and lang not in latent_langs
            )
        }

    return prompts


def lang_few_shot_prompts(
    df,
    tokenizer,
    langs,
    target_lang,
    latent_langs=None,
    lang_per_prompt=None,
    n_per_lang=1,
    num_prompts=200,
):
    if lang_per_prompt is None:
        lang_per_prompt = len(langs)
    if latent_langs is None:
        latent_langs = []
    prompts = []
    for _ in range(num_prompts):
        lang_sample = sample(langs, lang_per_prompt)
        concepts = df.sample(n_per_lang * lang_per_prompt)
        prompt = ""
        for i, lang in enumerate(lang_sample):
            for j in range(n_per_lang):
                row = concepts.iloc[i * n_per_lang + j]
                obj = row[f"senses_{lang}"][0]
                prompt += f"{obj}: {lang}\n"
        prompt += "_:"
        if len(tokenizer.encode("_:", add_special_tokens=False)) != 2:
            raise ValueError(
                "Weird tokenization going on, patchscope index might be wrong"
            )
        prompts.append(
            Prompt.from_strings(
                prompt, target_lang, {l: l for l in latent_langs}, tokenizer
            )
        )
    return prompts


class NotEnoughPromptsError(Exception):
    pass


def get_shifted_prompt_pairs(
    source_df,
    target_df,
    _source_prompts,
    _target_prompts,
    source_input_lang,
    source_output_lang,
    input_lang,
    target_lang,
    extra_langs,
    num_pairs,
    merge_extra_langs=True,
) -> tuple[list[Prompt], list[Prompt]]:
    check_source_tokens = (
        source_input_lang is not None or source_output_lang is not None
    )
    collected_pairs = 0
    source_prompts = []
    target_prompts = []
    source_target = list(product(source_df.iterrows(), target_df.iterrows()))
    shuffle(source_target)
    selected_source_rows = set()
    selected_target_rows = set()
    for (i, source_row), (j, target_row) in source_target:
        if source_row["word_original"] == target_row["word_original"]:
            continue
        src_p = deepcopy(_source_prompts[i])
        targ_p = deepcopy(_target_prompts[j])
        if check_source_tokens:
            targ_p.latent_tokens[f"src {source_output_lang}"] = src_p.target_tokens
            targ_p.latent_tokens[f"src {target_lang}"] = src_p.latent_tokens[
                target_lang
            ]
            targ_p.latent_strings[f"src {source_output_lang}"] = src_p.target_strings
            targ_p.latent_strings[f"src {target_lang}"] = src_p.latent_strings[
                target_lang
            ]
        for lang in extra_langs:
            if merge_extra_langs:
                targ_p.latent_tokens[f"src + tgt {lang}"] = ulist(
                    targ_p.latent_tokens[lang]
                    + (src_p.latent_tokens[lang] if check_source_tokens else [])
                )
                targ_p.latent_strings[f"src + tgt {lang}"] = ulist(
                    targ_p.latent_strings[lang]
                    + (src_p.latent_strings[lang] if check_source_tokens else [])
                )
            elif check_source_tokens:
                targ_p.latent_tokens[f"src {lang}"] = src_p.latent_tokens[lang]
                targ_p.latent_strings[f"src {lang}"] = src_p.latent_strings[lang]
        if targ_p.has_no_collisions(
            ignore_langs=extra_langs if merge_extra_langs else None
        ):
            source_prompts.append(src_p)
            target_prompts.append(targ_p)
            collected_pairs += 1
            selected_source_rows.add(i)
            selected_target_rows.add(j)
        if collected_pairs >= num_pairs:
            break
    if collected_pairs < num_pairs:
        print(
            f"Could only collect {collected_pairs} pairs for {source_input_lang} -> {source_output_lang} - {input_lang} -> {target_lang}, skipping..."
        )
        raise NotEnoughPromptsError
    print(
        f"Collected {collected_pairs} pairs for {source_input_lang} -> {source_output_lang} - {input_lang} -> {target_lang}"
        f" with {len(selected_source_rows)} source concepts and {len(selected_target_rows)} target concepts"
    )
    return source_prompts, target_prompts


def get_obj_id(sample_prompt, tokenizer):
    """
    For a prompt with the format '..."object" - X: "', return the index of the last token of the object.
    """
    split = sample_prompt.split('"')
    start = '"'.join(split[:-2])
    end = '"' + '"'.join(split[-2:])
    tok_start = tokenizer.encode(start, add_special_tokens=False)
    tok_end = tokenizer.encode(end, add_special_tokens=False)
    full = tokenizer.encode(sample_prompt, add_special_tokens=False)
    if tok_start + tok_end != full:
        raise ValueError(
            f"This is weird, check code, tokens don't match: {tokenizer.convert_ids_to_tokens(tok_start)} + {tokenizer.convert_ids_to_tokens(tok_end)} != {tokenizer.convert_ids_to_tokens(full)}"
        )
    idx = -len(tok_end) - 1
    return idx


def insert_between_chars(word, separator="-"):
    if isinstance(word, list):
        return [insert_between_chars(w, separator) for w in word]
    return separator.join(word)
