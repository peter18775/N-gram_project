import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
def load_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = sent_tokenize(text)
    tokenized_sentences = [["<s>"] + word_tokenize(s) + ["</s>"] for s in sentences]
    return tokenized_sentences

def build_vocab(tokenized_data, min_freq=1):
    
    counts = Counter(w for sent in tokenized_data for w in sent)
    vocab = {w for w, c in counts.items() if c >= min_freq}
    new_data = [[w if w in vocab else "<unk>" for w in sent] for sent in tokenized_data]
    vocab.add("<unk>")
    return new_data, sorted(vocab)

def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for word in vocab:
            file.write(f"{word}\n")
