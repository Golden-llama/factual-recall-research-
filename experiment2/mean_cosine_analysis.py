"""
mean_cosine_analysis.py -- Attribute/entity embedding alignment.

This computes:
  1. held-out composition accuracy
  2. mean cosine between each attribute-value token and the average embedding
     of entities sharing that attribute value

Toggle the settings below, or pass command-line arguments.

Examples:
    python3 mean_cosine_analysis.py
    python3 mean_cosine_analysis.py --entities 2000 --model disentangled
    python3 mean_cosine_analysis.py --entities 4000 --model both
"""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset_composition import COMPOSITION_ATTRS, ATTRIBUTE_SCHEMA, load_dataset
from tokenizer_composition import SROTokenizer
from train_composition import Config, load_model


# Manual toggles
ENTITY_COUNT = 4000
MODEL = "both"  # "summed", "disentangled", or "both"
CHECKPOINT = "model_best.pt"
OUT_DIR = "./outputs"
DATASET_PATH = "dataset_composition1000.json"
TOKENIZER_PATH = "tokenizer1000.json"
CSV_PATH = "mean_cosine_results.csv"


def token_ids_for_text(tokenizer, text):
    ids = tokenizer.encode(text)
    unk_id = getattr(tokenizer, "UNK_ID", 1)
    ids = [idx for idx in ids if idx != unk_id]
    if not ids:
        raise ValueError(f"No known tokenizer ids for: {text!r}")
    return ids


def embedding_for_text(embedding_weight, tokenizer, text, device):
    ids = token_ids_for_text(tokenizer, text)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return embedding_weight.index_select(0, ids).mean(dim=0)


def heldout_composition_accuracy(model, tokenizer, queries, device):
    heldout_queries = [
        q for q in queries
        if q.query_type == "composition" and q.held_out
    ]

    if not heldout_queries:
        raise ValueError("No held-out composition queries found.")

    correct = 0
    model.eval()

    with torch.no_grad():
        for q in heldout_queries:
            input_ids = tokenizer.encode(q.prompt, return_tensors="pt").to(device)
            logits, _ = model(input_ids)
            pred_id = logits[0, -1].argmax().item()
            gold_id = tokenizer.encode(q.answer)[0]
            correct += int(pred_id == gold_id)

    return correct / len(heldout_queries), len(heldout_queries)


def mean_attribute_cosine(model, tokenizer, entities, device):
    embedding_weight = model.embed.token_emb.weight.detach().to(device)

    cosines = []
    per_value_rows = []

    for attr in COMPOSITION_ATTRS:
        entities_by_value = defaultdict(list)
        for entity in entities:
            entities_by_value[entity[attr]].append(entity)

        for value in ATTRIBUTE_SCHEMA[attr]:
            group = entities_by_value.get(value, [])
            if not group:
                continue

            entity_vecs = [
                embedding_for_text(embedding_weight, tokenizer, entity["name"], device)
                for entity in group
            ]
            mean_entity_vec = torch.stack(entity_vecs).mean(dim=0)
            attr_value_vec = embedding_for_text(embedding_weight, tokenizer, value, device)

            cosine = F.cosine_similarity(
                mean_entity_vec.unsqueeze(0),
                attr_value_vec.unsqueeze(0),
            ).item()

            cosines.append(cosine)
            per_value_rows.append(
                {
                    "attribute": attr,
                    "value": value,
                    "n_entities": len(group),
                    "cosine": cosine,
                }
            )

    if not cosines:
        raise ValueError("No cosine values were computed.")

    return sum(cosines) / len(cosines), per_value_rows


def analyze_model(model_type, cfg, tokenizer, entities, test_queries, args, device):
    model_path = f"{args.out}/{args.entities}/{model_type}/{args.checkpoint}"
    print(f"\nLoading {model_type}: {model_path}")

    model = load_model(model_path, model_type, cfg, device)

    heldout_acc, n_heldout = heldout_composition_accuracy(
        model, tokenizer, test_queries, device
    )
    mean_cosine, per_value_rows = mean_attribute_cosine(
        model, tokenizer, entities, device
    )

    print(f"{model_type} held-out composition accuracy: {heldout_acc:.4f} ({n_heldout} queries)")
    print(f"{model_type} mean cosine: {mean_cosine:.6f}")

    return {
        "entities": args.entities,
        "model": model_type,
        "checkpoint": args.checkpoint,
        "heldout_acc": heldout_acc,
        "mean_cosine": mean_cosine,
        "n_heldout": n_heldout,
        "per_value_rows": per_value_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=ENTITY_COUNT)
    parser.add_argument("--model", choices=["summed", "disentangled", "both"], default=MODEL)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    tokenizer = SROTokenizer.load(args.tokenizer)
    cfg.vocab_size = tokenizer.vocab_size

    entities, train_queries, val_queries, test_queries = load_dataset(args.dataset)

    print(f"Entities label: {args.entities}")
    print(f"Dataset entities: {len(entities)}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Checkpoint: {args.checkpoint}")

    model_types = ["summed", "disentangled"] if args.model == "both" else [args.model]
    [
        analyze_model(model_type, cfg, tokenizer, entities, test_queries, args, device)
        for model_type in model_types
    ]


if __name__ == "__main__":
    main()
