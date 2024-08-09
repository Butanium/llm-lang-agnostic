This repo contains minimal code to reproduce results from {}

# How did we build the dataset
The code to rebuild the datasets is located in the `build_datasets` folder but are provided in the repo because you'd need to ask for extra Babelnet API credits / ask for the local index (and make them run 👻) which you don't want to.

## `word_translation.csv`
Those files are computed using the `gen_word_translation_dataset` function of `build_dataset/generate_dfs.py`.

For a lang $\ell$, we first generated a single word translation of `word_original`. 
This translated word is the first of each list in the $\ell$ column of the csv.

Then, using babelnet we find all meanings or "senses" related to this word, and then collect all the words or "lemma" that expresse those senses, for each language (including $\ell$)

## Disclaimer
For some languages we didn't compute the `word_translation.csv` file but you can only use them as output language