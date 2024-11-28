import re
import torch

def tokenize_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text.split()

def encode_text(tokenized_text, vocab, max_len):
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokenized_text]
    padded = encoded[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(encoded))
    return torch.tensor(padded, dtype=torch.long)