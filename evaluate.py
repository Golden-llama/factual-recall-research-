"""
evaluate.py — Evaluation suite.

Metrics:
  1. Attribute accuracy          — N×R accuracy broken down by relation type
  2. Compositional generalization — accuracy on held-out entity × composition pairs

Accuracy formula:  correct / (N × R)   where R = 11

For composition queries, multiple valid answers exist (any entity sharing the
same attribute value is a correct answer). Scoring checks whether the model's
top prediction is ANY valid answer, not just the one stored in the dataset.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from tokenizer import SROTokenizer
import json
import argparse

from train   import Config, load_model
from dataset import (load_dataset, EXTRACTION_RELATIONS,
                     COMPOSITION_RELATIONS, ALL_RELATIONS,
                     ATTRIBUTE_SCHEMA, COMPOSITION_ATTRS)


# ═══════════════════════════════════════════════════════════════
# Build valid answer sets for composition queries
# ═══════════════════════════════════════════════════════════════

def build_valid_answers(entities):
    """
    For each (subject, composition_relation) pair, return the set of ALL
    valid answer entity names — any entity sharing the same attribute value.

    e.g. Bliblax-210 → same_color_as → {all entities with color=green} - {Bliblax-210}

    Extraction queries have exactly one correct answer so they don't need this.
    """
    # Index: (attr, value) → set of entity names with that value
    value_to_names = defaultdict(set)
    for e in entities:
        for attr in COMPOSITION_ATTRS:
            value_to_names[(attr, e[attr])].add(e["name"])

    # Build valid answer sets for each (subject, relation) pair
    valid = {}
    for e in entities:
        for attr in COMPOSITION_ATTRS:
            rel     = f"same_{attr}_as"
            val     = e[attr]
            # All entities sharing this value, excluding the subject itself
            answers = value_to_names[(attr, val)] - {e["name"]}
            valid[(e["name"], rel)] = answers

    return valid


# ═══════════════════════════════════════════════════════════════
# Core scoring functions
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def score_extraction(model, tokenizer, query, device):
    """
    Score an extraction query. Exactly one correct answer.
    Returns True if the model's top prediction matches the answer.
    """
    input_ids = tokenizer.encode(query.prompt, return_tensors="pt").to(device)
    if input_ids.size(1) >= 512:
        return None

    logits, _ = model(input_ids)
    probs     = F.softmax(logits[0, -1, :], dim=-1)
    ans_tok   = tokenizer.encode(query.answer)[0]
    rank      = (probs > probs[ans_tok]).sum().item() + 1
    return rank == 1


@torch.no_grad()
def score_composition(model, tokenizer, query, valid_answers, device):
    """
    Score a composition query against ALL valid answers.
    Correct if the model's top-1 prediction is any entity sharing
    the same attribute value as the subject.

    valid_answers: set of entity name strings that are all correct.
    """
    input_ids = tokenizer.encode(query.prompt, return_tensors="pt").to(device)
    if input_ids.size(1) >= 512:
        return None, None

    logits, _ = model(input_ids)
    probs     = F.softmax(logits[0, -1, :], dim=-1)

    # Top predicted token
    top_tok      = probs.argmax().item()
    top_word     = tokenizer.convert_ids_to_tokens(top_tok)

    # Correct if top prediction is any valid answer
    correct = top_word in valid_answers
    return correct, top_word


# ═══════════════════════════════════════════════════════════════
# 1. N × R Attribute Accuracy
# ═══════════════════════════════════════════════════════════════

def eval_NxR_accuracy(model, tokenizer, test_queries, valid_answers_map, device):
    """
    Accuracy = correct / (N × R)
    Extraction:  single correct answer
    Composition: any entity sharing the same attribute value is correct
    """
    model.eval()

    by_relation   = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type       = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct, total = 0, 0

    for q in test_queries:
        if q.query_type == "extraction":
            correct = score_extraction(model, tokenizer, q, device)
        else:
            valid   = valid_answers_map.get((q.subject, q.relation), set())
            correct, _ = score_composition(model, tokenizer, q, valid, device)

        if correct is None:
            continue

        correct = int(correct)
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


# ═══════════════════════════════════════════════════════════════
# 2. Compositional Generalization
# ═══════════════════════════════════════════════════════════════

def eval_compositional_generalization(model, tokenizer, test_queries,
                                       held_out_names, valid_answers_map, device):
    """
    Accuracy on composition queries for held-out entities.
    Uses multi-answer scoring — any entity sharing the attribute is correct.
    """
    model.eval()
    held_comp = [
        q for q in test_queries
        if q.query_type == "composition" and q.subject in held_out_names
    ]

    by_relation = defaultdict(lambda: {"correct": 0, "total": 0})
    correct, total = 0, 0

    for q in held_comp:
        valid      = valid_answers_map.get((q.subject, q.relation), set())
        c, top_pred = score_composition(model, tokenizer, q, valid, device)
        if c is None:
            continue
        c = int(c)
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


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════

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
    print(f"   (correct = model's top prediction is ANY entity sharing the attribute)")
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


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summed",       required=True)
    parser.add_argument("--disentangled", required=True)
    parser.add_argument("--dataset",      default="dataset.json")
    parser.add_argument("--out",          default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = Config()

    print("Loading dataset...")
    entities, entity_index, train_queries, test_queries, held_out_names = \
        load_dataset(args.dataset)

    tokenizer      = SROTokenizer.from_dataset(args.dataset)
    cfg.vocab_size = tokenizer.vocab_size

    # Build valid answer sets for composition queries
    print("Building valid answer sets for composition queries...")
    valid_answers_map = build_valid_answers(entities)

    # Sanity check — show how many valid answers exist per relation
    sample_entity = entities[0]
    print(f"\n  Valid answer counts for '{sample_entity['name']}':")
    for attr in COMPOSITION_ATTRS:
        rel   = f"same_{attr}_as"
        valid = valid_answers_map.get((sample_entity["name"], rel), set())
        print(f"    {rel:<22} → {len(valid)} valid answers  "
              f"(attr={sample_entity[attr]})")

    print("\nLoading models...")
    summed_model = load_model(args.summed,       "summed",       cfg, device)
    disent_model = load_model(args.disentangled, "disentangled", cfg, device)

    all_results = {}
    for model_name, model in [("summed", summed_model), ("disentangled", disent_model)]:
        print(f"\n{'─'*50}\nEvaluating: {model_name}\n{'─'*50}")

        print("① N×R accuracy...")
        attr_acc = eval_NxR_accuracy(
            model, tokenizer, test_queries, valid_answers_map, device)

        print("② Compositional generalization...")
        comp_gen = eval_compositional_generalization(
            model, tokenizer, test_queries, held_out_names, valid_answers_map, device)

        results = {"attr_acc": attr_acc, "comp_gen": comp_gen}
        all_results[model_name] = results
        print_report(model_name, results)

    print_comparison(all_results["summed"], all_results["disentangled"])

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.out}")