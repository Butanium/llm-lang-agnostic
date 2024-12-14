import json
import re
import numpy as np
import numpy as np
import scipy.stats as stats

def ulist(lst):
    """
    Returns a list with unique elements from the input list.
    """
    return list(dict.fromkeys(lst))

SPACE_TOKENS = ["▁", "Ġ", " "]


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


def add_spaces(tokens):
    return sum([[s + token for token in tokens] for s in SPACE_TOKENS], []) + tokens


# TODO?: Add capitalization


def byte_string_to_list(input_string):
    # Find all parts of the string: either substrings or hex codes
    parts = re.split(r"(\\x[0-9a-fA-F]{2})", input_string)
    result = []
    for part in parts:
        if re.match(r"\\x[0-9a-fA-F]{2}", part):
            # Convert hex code to integer
            result.append(int(part[2:], 16))
        else:
            if part:  # Skip empty strings
                result.append(part)
    return result


def unicode_prefixes(tok_str):
    encoded = str(tok_str.encode())[2:-1]
    if "\\x" not in encoded:
        return []  # No bytes in the string
    chr_list = byte_string_to_list(encoded)
    if isinstance(chr_list[0], int):
        first_byte_token = [
            f"<{hex(chr_list[0]).upper()}>"
        ]  # For llama2 like tokenizer, this is how bytes are represented
    else:
        first_byte_token = []
    # We need to convert back to latin1 to get the character
    for i, b in enumerate(chr_list):
        # those bytes are not valid latin1 characters and are shifted by 162 in Llama3 and Qwen
        if isinstance(b, str):
            continue
        if b >= 127 and b <= 160:
            chr_list[i] += 162
        chr_list[i] = chr(chr_list[i])
    # Convert back to string
    vocab_str = "".join(
        chr_list
    )  # This is the string that will be in the tokenizer vocab for Qwen and Llama3
    return first_byte_token + token_prefixes(vocab_str)


def process_tokens(words: str | list[str], tok_vocab):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        with_prefixes = token_prefixes(word) + unicode_prefixes(word)
        with_spaces = add_spaces(with_prefixes)
        for word in with_spaces:
            if word in tok_vocab:
                final_tokens.append(tok_vocab[word])
    return ulist(final_tokens)


def process_tokens_with_tokenization(
    words: str | list[str], tokenizer, i_am_hacky=False
):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        if i_am_hacky:
            hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
            length = len(hacky_token)
            tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if tokens[:length] != hacky_token:
                raise ValueError(
                    "I didn't expect this to happen, please check this code"
                )
            if len(tokens) > length:
                final_tokens.append(tokens[length])
        else:
            # Assuming the tokenizer was initialized with add_prefix_space=False
            token = tokenizer(word, add_special_tokens=False).input_ids[0]
            token_with_start_of_word = tokenizer(
                " " + word, add_special_tokens=False
            ).input_ids[0]
            if token == token_with_start_of_word:
                raise ValueError(
                    "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False. Not good :("
                )
            final_tokens.append(token)
            if (
                token_with_start_of_word
                != tokenizer(" ", add_special_tokens=False).input_ids[0]
            ):
                final_tokens.append(token_with_start_of_word)
    return ulist(final_tokens)


def mean_no_none(lst):
    filtered = [x for x in lst if x is not None]
    if len(filtered) == 0:
        return None
    return np.mean(filtered)


def ci_no_none(lst, confidence=0.95):
    filtered = [x for x in lst if x is not None]
    if len(filtered) == 0:
        return None
    sem = stats.sem(filtered)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., len(filtered) - 1)  # Confidence interval
    return h


def load_dict(file):
    with open(file, "r") as f:
        json_dic = json.load(f)
    return json_dic