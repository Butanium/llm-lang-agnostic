This repo contains minimal code to reproduce results from [Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](https://arxiv.org/abs/2411.08745)
 previously published under the title [How Do Llamas Process Multilingual Text? A Latent Exploration through Activation Patching](https://openreview.net/forum?id=0ku2hIm4BS), spotlight at the ICML 2024 Mechanistic interpretability workshop. 
# Setup
Create a python environment and `pip install -r requirements.txt`. We think the code probably work with `python>=3.7` but only tested it with `python=3.11.x`.

If this happens to not work you can use the versions specified in `pip.freeze` which contains all the package we have installed with their version in our local `conda` environment.

# Reproducing results
```bash	
chmod +x compute_results.sh
./compute_results.sh
```

# Datasets used in the paper
The code to rebuild the datasets is located in the `build_datasets` folder but are provided in the repo because you'd need to ask for extra Babelnet API credits / ask for the local index (and make them run 👻) which you don't want to. If you want to do so, make sure to install `babelnet` using:
```bash
pip install --ignore-requires-python babelnet cache_decorator emoji
```

## `word_translation.csv`
Those files are computed using the `main_translation_dataset` function of `build_dataset/build_bn_dataset.py`.

For a lang $\ell$, we first generated a single word translation of `word_original`. 
This translated word is the first of each list in the $\ell$ column of the csv.

Then, using babelnet we find all meanings or "senses" related to this word, and then collect all the words or "lemma" that expresse those senses, for each language (including $\ell$)

**Disclaimer:** For some languages we didn't compute the `word_translation.csv` file so you can only use them as output language. If you need one of those, shoot us an email and we'll add it.

# Extra datasets that did not make it yet in the paper
## `synset_dataset.csv`
Those files are computed using the `main_synset_dataset` function of `build_dataset/build_bn_dataset.py`.

We took 200 words from https://en.wiktionary.org/wiki/Appendix:Basic_English_word_list#Things_-_200_picturable_words and found their canonical concept or "synset" in babelnet.

## `cloze_dataset.csv`
Those files are computed using the `main_cloze_dataset` function of `build_dataset/build_bn_dataset.py`.

We took the synsets from `synset_dataset.csv` and for each language we collected the different definitions available in babelnet. 

The dataset contains several columns:
- `original_definitions` (`str`): the original definitions from babelnet
- `clozes` (`cloze: str, acceptable_sense: tuple[str]`): definitions where we replaced one of the sense by a placeholder `____`. The `acceptable_sense` is the list of senses that don't appear in the cloze.
- `clozes_with_start_of_word` (`cloze: str, acceptable_sense: tuple[str]`): same as `clozes` but we search both for the sense alone and for words that start with the sense.
- `definitions_wo_ref` (`str`): definitions without the reference to any sense. 
