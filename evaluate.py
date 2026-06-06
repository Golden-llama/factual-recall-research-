"""
evaluate.py — Evaluation suite.

Metrics:
  1. Attribute accuracy          — N×R accuracy broken down by relation type
  2. Compositional generalization — accuracy on held-out entity × composition pairs

Accuracy formula:  correct / (N × R)   where R = 11
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from tokenizer import SROTokenizer
import json
import argparse

from train   import Config, load_model
from dataset import (load_dataset, EXTRACTION_RELATIONS,
                     COMPOSITION_RELATIONS, ALL_RELATIONS)


# Core scoring function

@torch.no_grad()
def score_query(model, tokenizer, query, device):
    """
    Returns rank of correct answer token at the <O> position.
    Rank 1 = correct (top prediction).
    """
    input_ids = tokenizer.encode(query.prompt, return_tensors="pt").to(device)
    if input_ids.size(1) >= 512:
        return None

    logits, _ = model(input_ids)
    probs     = F.softmax(logits[0, -1, :], dim=-1)
    ans_tok   = tokenizer.encode(query.answer)[0]
    rank      = (probs > probs[ans_tok]).sum().item() + 1
    return rank


# 1. N × R Attribute Accuracy

def eval_NxR_accuracy(model, tokenizer, test_queries, device):
    """
    Accuracy = correct / (N × R)
    Broken down by relation type and extraction vs composition.
    """
    model.eval()

    by_relation = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type     = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct, total = 0, 0

    for q in test_queries:
        rank = score_query(model, tokenizer, q, device)
        if rank is None:
            continue

        correct = int(rank == 1)
        by_relation[q.relation]["correct"] += correct
        by_relation[q.relation]["total"]   += 1
        by_type[q.query_type]["correct"]   += correct
        by_type[q.query_type]["total"]     += 1
        total_correct += correct
        total         += 1

    return {
        "overall":     total_correct / total if total else 0,
        "N_times_R":   total,
        "by_relation": {
            rel: v["correct"] / v["total"]
            for rel, v in by_relation.items() if v["total"] > 0
        },
        "by_type": {
            t: v["correct"] / v["total"]
            for t, v in by_type.items() if v["total"] > 0
        },
    }


# 2. Compositional Generalization

def eval_compositional_generalization(model, tokenizer, test_queries, held_out_names, device):
    """
    Accuracy on composition queries for held-out entities.
    These entities had composition queries withheld from training —
    the model must apply the relation to a subject it never composed before.
    """
    model.eval()
    held_comp = [
        q for q in test_queries
        if q.query_type == "composition" and q.subject in held_out_names
    ]

    by_relation = defaultdict(lambda: {"correct": 0, "total": 0})
    correct, total = 0, 0

    for q in held_comp:
        rank = score_query(model, tokenizer, q, device)
        if rank is None:
            continue
        c = int(rank == 1)
        correct += c
        total   += 1
        by_relation[q.relation]["correct"] += c
        by_relation[q.relation]["total"]   += 1

    return {
        "P@1": correct / total if total else 0,
        "n":   total,
        "by_relation": {
            r: v["correct"] / v["total"]
            for r, v in by_relation.items() if v["total"] > 0
        },
    }


# Reporting

def print_report(name, r):
    W = 60
    print(f"\n{'═'*W}")
    print(f" {name}")
    print(f"{'═'*W}")

    a = r["attr_acc"]
    print(f"\n① N×R Accuracy  (n={a['N_times_R']})")
    print(f"   Overall:      {a['overall']:.4f}")
    print(f"   Extraction:   {a['by_type'].get('extraction',  0):.4f}")
    print(f"   Composition:  {a['by_type'].get('composition', 0):.4f}")
    print(f"\n   By relation:")
    for rel in ALL_RELATIONS:
        acc = a["by_relation"].get(rel, 0)
        tag = "[comp]" if rel in COMPOSITION_RELATIONS else "      "
        bar = "█" * int(acc * 25)
        print(f"   {tag} {rel:<22} {acc:.3f} {bar}")

    g = r["comp_gen"]
    print(f"\n② Compositional Generalization  (n={g['n']})")
    print(f"   Overall P@1: {g['P@1']:.4f}")
    print(f"\n   By relation:")
    for rel, acc in g["by_relation"].items():
        bar = "█" * int(acc * 25)
        print(f"   {rel:<25} {acc:.3f} {bar}")


def print_comparison(sr, cr):
    W = 65
    print(f"\n{'═'*W}")
    print(" COMPARISON: Summed (baseline) vs Disentangled (method)")
    print(f"{'═'*W}")

    def row(label, s, c):
        delta = c - s
        arrow = "↑ method" if delta > 0 else "↓ method"
        print(f"  {label:<40} summed={s:.4f}  disentangled={c:.4f}  {arrow} ({delta:+.4f})")

    print()
    row("① Overall N×R accuracy",
        sr["attr_acc"]["overall"],
        cr["attr_acc"]["overall"])
    row("   Extraction accuracy",
        sr["attr_acc"]["by_type"].get("extraction",  0),
        cr["attr_acc"]["by_type"].get("extraction",  0))
    row("   Composition accuracy",
        sr["attr_acc"]["by_type"].get("composition", 0),
        cr["attr_acc"]["by_type"].get("composition", 0))

    print()
    row("② Comp. generalization P@1 (held-out entities)",
        sr["comp_gen"]["P@1"],
        cr["comp_gen"]["P@1"])

    print(f"\n   By relation:")
    all_rels = set(sr["comp_gen"]["by_relation"]) | set(cr["comp_gen"]["by_relation"])
    for rel in sorted(all_rels):
        s = sr["comp_gen"]["by_relation"].get(rel, 0)
        c = cr["comp_gen"]["by_relation"].get(rel, 0)
        arrow = "↑" if c > s else "↓"
        print(f"   {rel:<25} summed={s:.3f}  disentangled={c:.3f}  {arrow} ({c-s:+.3f})")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summed",       required=True)
    parser.add_argument("--disentangled", required=True)
    parser.add_argument("--dataset",      default="dataset.json")
    parser.add_argument("--out",          default="eval_results.json")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg       = Config()

    print("Loading dataset...")
    entities, entity_index, train_queries, test_queries, held_out_names = \
        load_dataset(args.dataset)

    tokenizer      = SROTokenizer.from_dataset(args.dataset)
    cfg.vocab_size = tokenizer.vocab_size

    print("Loading models...")
    summed_model = load_model(args.summed,       "summed",       cfg, device)
    disent_model = load_model(args.disentangled, "disentangled", cfg, device)

    all_results = {}
    for model_name, model in [("summed", summed_model), ("disentangled", disent_model)]:
        print(f"\n{'─'*50}\nEvaluating: {model_name}\n{'─'*50}")

        print("① N×R accuracy...")
        attr_acc = eval_NxR_accuracy(model, tokenizer, test_queries, device)

        print("② Compositional generalization...")
        comp_gen = eval_compositional_generalization(
            model, tokenizer, test_queries, held_out_names, device)

        results = {"attr_acc": attr_acc, "comp_gen": comp_gen}
        all_results[model_name] = results
        print_report(model_name, results)

    print_comparison(all_results["summed"], all_results["disentangled"])

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.out}")