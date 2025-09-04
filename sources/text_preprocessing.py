import string
import re
from typing import Dict, List

def tokenize_long_text(
    text: str, 
    vocab: Dict[str, int], 
    output_sequence_length: int
) -> List[List[int]]:
    """
    Tokenizes text, strips punctuation, splits into fixed-length blocks, and converts words to dictionary-based indices.

    Args:
    text (str): The original text to tokenize.
    vocab (Dict[str, int]): A dictionary where keys are words and values are indices.
    output_sequence_length (int): The length of the token block

    Returns:
    List[List[int]]: A list of blocks, each block a list of word indices of length output_sequence_length.
    Empty positions are filled with the "<PAD>" index, unknown words are filled with "<UNK>".
    """
    tokens = re.sub(f"([{string.punctuation}])", r'', text).lower().split()
    blocks = []
    for i in range(0, len(tokens), output_sequence_length):
        block_tokens = tokens[i:i+output_sequence_length]
        block_int = [vocab["<PAD>"]] * output_sequence_length
        for j in range(len(block_tokens)):
            block_int[j] = vocab.get(block_tokens[j], vocab["<UNK>"])
        blocks.append(block_int)
    return blocks