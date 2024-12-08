import pandas as pd
import ast
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def load_cloze(lang):
    path = DATA_PATH / lang / "cloze_dataset.csv"
    return pd.read_csv(path)


def get_word_translation_dataset(
    source_lang: str,
    target_langs: str | list[str],
    num_words: None | int = None,
    do_sample=True,
    v2=True,
):
    num = "2" if v2 else ""
    if isinstance(target_langs, str):
        target_langs = [target_langs]
    df = pd.read_csv(DATA_PATH / source_lang / f"word_translation{num}.csv")
    if num_words is not None:
        if do_sample:
            df = df.sample(num_words)
        else:
            df = df.head(num_words)

    out_df = pd.DataFrame()
    for lang in target_langs:
        out_df[lang] = df[lang].map(ast.literal_eval)
    out_df[source_lang] = df[source_lang].map(ast.literal_eval)
    assert (
        out_df.map(lambda x: x != []).all(axis=1)
    ).all(), "Some translations are empty"
    out_df["word_original"] = df["word_original"]
    return out_df


def get_cloze_dataset(
    langs: str | list[str],
    num_words: None | int = None,
    do_sample=True,
    drop_no_defs=False,
):
    if isinstance(langs, str):
        langs = [langs]
    dfs = {}
    for lang in langs:
        df = load_cloze(lang)
        to_eval = [
            "definitions_wo_ref",
            "senses",
            "original_definitions",
            "clozes",
            "clozes_with_start_of_word",
        ]
        for col in to_eval:
            df[col] = df[col].map(eval)
        if drop_no_defs:
            df = df[df["definitions_wo_ref"].map(lambda x: x != [])]
        if num_words is not None:
            if do_sample:
                df = df.sample(num_words)
            else:
                df = df.head(num_words)
        dfs[lang] = df
    # Join on synset
    merged_df: pd.DataFrame = dfs[langs[0]]
    original_cols = merged_df.columns
    for lang in langs[1:]:
        merged_df = merged_df.merge(
            dfs[lang],
            on=["synset", "word_original"],
            how="inner",
            suffixes=("", f"_{lang}"),
        )
    # add _lang suffix to columns
    merged_df = merged_df.rename(
        columns={
            col: f"{col}_{langs[0]}" if col not in ["word_original", "synset"] else col
            for col in original_cols
        }
    )
    return merged_df
