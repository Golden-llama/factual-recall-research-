"""
train_experiment.py -- Wires dataset + tokenizer + training loop together.

Usage:
    python3 train_experiment.py --model both --entities 1000
    python3 train_experiment.py --model both --entities 2000
    python3 train_experiment.py --model both --entities 4000
"""

import argparse
from collections import defaultdict

import torch

from dataset_composition import (
    ATTRIBUTE_SCHEMA,
    COMPOSITION_ATTRS,
    get_dataloaders,
    load_dataset,
)
from tokenizer_composition import SROTokenizer
from train_composition import Config, load_model, train


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


@torch.no_grad()
def heldout_composition_accuracy(model, tokenizer, queries, device):
    heldout_queries = [
        q for q in queries
        if q.query_type == "composition" and q.held_out
    ]
    if not heldout_queries:
        raise ValueError("No held-out composition queries found.")

    correct = 0
    model.eval()
    for q in heldout_queries:
        input_ids = tokenizer.encode(q.prompt, return_tensors="pt").to(device)
        logits, _ = model(input_ids)
        pred_id = logits[0, -1].argmax().item()
        gold_id = tokenizer.encode(q.answer)[0]
        correct += int(pred_id == gold_id)

    return correct / len(heldout_queries), len(heldout_queries)


@torch.no_grad()
def attribute_group_norm_stats(model, tokenizer, entities, device):
    embedding_weight = model.embed.token_emb.weight.detach().to(device)

    subject_vecs = [
        embedding_for_text(embedding_weight, tokenizer, entity["name"], device)
        for entity in entities
    ]
    avg_subject_norm = torch.stack([vec.norm() for vec in subject_vecs]).mean().item()

    group_norms = []
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
            group_norms.append(mean_entity_vec.norm().item())

    if not group_norms:
        raise ValueError("No group norms were computed.")

    mean_group_norm = sum(group_norms) / len(group_norms)
    ratio = mean_group_norm / avg_subject_norm
    return mean_group_norm, avg_subject_norm, ratio


def analyze_saved_model(model_type, cfg, tokenizer, entities, test_queries, device, model_path):
    print(f"\nLoading trained {model_type} model for analysis: {model_path}")
    model = load_model(model_path, model_type, cfg, device)

    heldout_acc, n_heldout = heldout_composition_accuracy(
        model, tokenizer, test_queries, device
    )
    mean_group_norm, avg_subject_norm, ratio = attribute_group_norm_stats(
        model, tokenizer, entities, device
    )

    print(f"{model_type} held-out composition accuracy: {heldout_acc:.4f} ({n_heldout} queries)")
    print(f"{model_type} mean group-average norm: {mean_group_norm:.6f}")
    print(f"{model_type} average subject embedding norm: {avg_subject_norm:.6f}")
    print(f"{model_type} group norm / average subject norm: {ratio:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["summed", "disentangled", "both"], default="both")
    parser.add_argument("--out", default="./10yes10no/outputs")
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Kept for compatibility; this script currently loads existing dataset/tokenizer files.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every

    tokenizer = SROTokenizer.load("tokenizer1000.json")
    cfg.vocab_size = tokenizer.vocab_size
    entities, train_queries, val_queries, test_queries = load_dataset("dataset_composition1000.json")

    print(f"Entities label for output folder: {args.entities}")
    print(f"Dataset entities: {len(entities)}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    print("\nBuilding dataloaders...")
    train_dl = get_dataloaders(train_queries, tokenizer, batch_size=cfg.batch_size)
    unseen_dl = get_dataloaders(val_queries, tokenizer, batch_size=cfg.batch_size)

    entity_out = f"{args.out}/{args.entities}"

    if args.model in ("summed", "both"):
        print(f"\n{'=' * 55}")
        print("Training: SUMMED (baseline)")
        print(f"{'=' * 55}")
        train(
            "summed",
            cfg,
            train_dl,
            device,
            out_dir=f"{entity_out}/summed",
            unseen_dl=unseen_dl,
            train_eval_dl=train_dl,
        )
        analyze_saved_model(
            "summed",
            cfg,
            tokenizer,
            entities,
            test_queries,
            device,
            model_path=f"{entity_out}/summed/{args.checkpoint}",
        )

    if args.model in ("disentangled", "both"):
        print(f"\n{'=' * 55}")
        print("Training: DISENTANGLED (method)")
        print(f"{'=' * 55}")
        train(
            "disentangled",
            cfg,
            train_dl,
            device,
            out_dir=f"{entity_out}/disentangled",
            unseen_dl=unseen_dl,
            train_eval_dl=train_dl,
        )
        analyze_saved_model(
            "disentangled",
            cfg,
            tokenizer,
            entities,
            test_queries,
            device,
            model_path=f"{entity_out}/disentangled/{args.checkpoint}",
        )

    print("\nTraining complete.")
