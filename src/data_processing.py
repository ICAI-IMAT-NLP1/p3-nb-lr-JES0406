from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    with open(infile, "r") as f:
        lines = f.readlines()
    
    lines_splited = [line.split("\t") for line in lines]
    pairs = {" ".join(line[:-1]): line[-1] for line in lines_splited} # This removes the \t from the sentences, but it is ok

    proces_words = lambda x: tokenize(x)
    examples: List[SentimentExample] = [SentimentExample(proces_words(words), int(label)) for words, label in pairs.items()]
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    words_lists = [example.words for example in examples]
    u_words = set()
    for words in words_lists:
        u_words.update(words)
    vocab: Dict[str, int] = {word: i for i, word in enumerate(list(u_words))}

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    word_counts = Counter(text) if not binary else Counter(list(set(text)))
    return torch.tensor([word_counts[word] for word in vocab])
