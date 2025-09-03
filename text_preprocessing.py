import string
import re

def tokenize_long_text(text, vocab, output_sequence_length):
    tokens = re.sub(f"([{string.punctuation}])", r'', text).lower().split()
    blocks = []
    for i in range(0, len(tokens), output_sequence_length):
        block_tokens = tokens[i:i+output_sequence_length]
        block_int = [vocab["<PAD>"]] * output_sequence_length
        for j in range(len(block_tokens)):
            block_int[j] = vocab.get(block_tokens[j], vocab["<UNK>"])
        blocks.append(block_int)
    return blocks