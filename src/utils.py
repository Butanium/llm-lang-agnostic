from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer


def ulist(lst):
    """
    Returns a list with unique elements from the input list.
    """
    return list(dict.fromkeys(lst))


def get_tokenizer(model_or_tokenizer):
    """
    Returns the tokenizer of the given model or the given tokenizer.
    """
    if isinstance(model_or_tokenizer, LanguageModel) or isinstance(
        model_or_tokenizer, UnifiedTransformer
    ):
        return model_or_tokenizer.tokenizer
    return model_or_tokenizer
