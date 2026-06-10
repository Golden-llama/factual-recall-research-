"""
tokenizer.py — Minimal custom tokenizer for the synthetic entity experiment.

Vocabulary (1052 tokens total):
  9    special tokens  (<PAD>, <UNK>, <S>, </S>, <R>, </R>, <O>, </O>, <|sep|>)
  35   attribute values (blue, triangle, metal, ...)
  8    relation words   (color, shape, same, as, ...)
  1000 entity names    (loaded from dataset.json)

Tokenization strategy:
  Each token in the vocabulary maps to exactly one integer id.
  Entity names and attribute values are single tokens (no subword splitting).
  Relation strings like "same color as" split on whitespace into ["same","color","as"].
  Unknown tokens map to <UNK>.

Usage:
    from tokenizer import SROTokenizer

    # Build from saved dataset (recommended — entity names match exactly)
    tokenizer = SROTokenizer.from_dataset("dataset.json")

    ids    = tokenizer.encode("<S> Zorblax-7 </S> <R> color </R> <O> blue </O>")
    tokens = tokenizer.decode(ids)
    print(tokenizer.vocab_size)   # 1052
"""

from dataset import ATTRIBUTE_SCHEMA, EXTRACTION_RELATIONS, COMPOSITION_RELATIONS
from typing import List, Union
import json


class SROTokenizer:
    """
    Simple whitespace tokenizer whose full vocabulary is known at build time.
    Every entity name and attribute value is a single token.
    Relation strings tokenize as individual words.
    """

    SPECIAL_TOKENS = [
        "<PAD>",      # 0  — padding
        "<UNK>",      # 1  — unknown token
        "<S>",        # 2  — open subject
        "</S>",       # 3  — close subject
        "<R>",        # 4  — open relation
        "</R>",       # 5  — close relation
        "<O>",        # 6  — open object
        "</O>",       # 7  — close object
    ]

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, entity_names: List[str]):
        """
        Build vocabulary from a list of entity name strings.
        Call SROTokenizer.from_dataset() to load names from dataset.json.

        Vocabulary order (deterministic):
            specials → attribute values → relation words → entity names
            (all sorted alphabetically within each group)
        """
        attr_values  = sorted(set(v for vals in ATTRIBUTE_SCHEMA.values() for v in vals))
        rel_words    = sorted(set(
            w for rel in EXTRACTION_RELATIONS + COMPOSITION_RELATIONS
            for w in rel.replace("_", " ").split()
        ))
        entity_names_sorted = sorted(set(entity_names))

        vocab = (
            self.SPECIAL_TOKENS +
            attr_values         +
            rel_words           +
            entity_names_sorted
        )

        self.token2id   = {tok: i for i, tok in enumerate(vocab)}
        self.id2token   = {i: tok for i, tok in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"SROTokenizer built:")
        print(f"  Special tokens:   {len(self.SPECIAL_TOKENS)}")
        print(f"  Attribute values: {len(attr_values)}")
        print(f"  Relation words:   {len(rel_words)}")
        print(f"  Entity names:     {len(entity_names_sorted)}  (from dataset.json)")
        print(f"  Total vocab:      {self.vocab_size}")

    @classmethod
    def from_dataset(cls, dataset_path: str = "dataset.json") -> "SROTokenizer":
        """
        Build tokenizer using the entity names already saved in dataset.json.
        This is the recommended way to instantiate — guarantees the tokenizer
        vocab matches the dataset exactly.

        Run dataset.py first to generate dataset.json.
        """
        with open(dataset_path) as f:
            data = json.load(f)
        entity_names = [e["name"] for e in data["entities"]]
        print(f"  Loaded {len(entity_names)} entity names from {dataset_path}")
        return cls(entity_names)

    # Core tokenization
   

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens.
        Special tokens are padded with spaces first so they never get
        merged with adjacent characters during the whitespace split.
        """
        for special in self.SPECIAL_TOKENS:
            text = text.replace(special, f" {special} ")
        return text.split()

    def encode(self, text: str, return_tensors: str = None) -> Union[List[int], "torch.Tensor"]:
        """
        Convert a string to a list of integer ids.
        Unknown tokens map to UNK_ID (should never happen with synthetic data).
        Pass return_tensors="pt" to get a (1, T) LongTensor.
        """
        ids = [self.token2id.get(tok, self.UNK_ID) for tok in self.tokenize(text)]
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert a list of integer ids back to a string."""
        tokens = [self.id2token.get(i, "<UNK>") for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return " ".join(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token2id.get(token, self.UNK_ID)

    def convert_ids_to_tokens(self, id: int) -> str:
        return self.id2token.get(id, "<UNK>")

    # Save / load

    def save(self, path: str = "tokenizer.json"):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)
        print(f"  Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str = "tokenizer.json") -> "SROTokenizer":
        """
        Load a previously saved tokenizer directly from tokenizer.json.
        Faster than rebuilding from dataset.json if the vocab hasn't changed.
        """
        tok = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        tok.token2id   = data["token2id"]
        tok.id2token   = {int(i): t for t, i in tok.token2id.items()}
        tok.vocab_size = len(tok.token2id)
        print(f"  Tokenizer loaded from {path}  (vocab size: {tok.vocab_size})")
        return tok


# Inspection

if __name__ == "__main__":
    import os

    if not os.path.exists("dataset.json"):
        print("dataset.json not found — run dataset.py first.")
        exit(1)

    tokenizer = SROTokenizer.from_dataset("dataset.json")
    tokenizer.save("tokenizer.json")

    print()
    test_sequences = [
        "<S> Bliblax-210 </S> <R> color </R> <O> blue </O>",
        "<S> Bliforn-232 </S> <R> same color as </R> <O> Blimpnik-34 </O>",
        "<S> Bligast-300 </S> <R> material </R> <O> metal </O>",
    ]

    print("── Encoding examples ──")
    for seq in test_sequences:
        ids    = tokenizer.encode(seq)
        tokens = tokenizer.tokenize(seq)
        print(f"\n  Input:   {seq}")
        print(f"  Tokens:  {tokens}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {tokenizer.decode(ids)}")

    print(f"\n── Special token IDs ──")
    for tok in SROTokenizer.SPECIAL_TOKENS:
        print(f"  {tok:<12} → {tokenizer.convert_tokens_to_ids(tok)}")

    print(f"\n── Vocab size: {tokenizer.vocab_size} ──")
    print(f"   (vs GPT-2: 50,264  —  {50264 // tokenizer.vocab_size}× smaller)")