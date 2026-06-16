import torch
from collections import defaultdict
from tokenizer_composition import SROTokenizer
from train_composition import Config, load_model
from dataset_composition import (
    load_dataset,
    EXTRACTION_RELATIONS,
    COMPOSITION_RELATIONS,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg    = Config()

entities, entity_index, train_queries, val_queries, test_queries, held_out_names = \
    load_dataset("dataset_composition1000.json")

tokenizer      = SROTokenizer.load("tokenizer1000.json")
cfg.vocab_size = tokenizer.vocab_size

summed_model = load_model("outputs/summed/model_best.pt",       "summed",       cfg, device)
disent_model = load_model("outputs/disentangled/model_best.pt", "disentangled", cfg, device)


@torch.no_grad()
def decode_answer(model, input_ids):
    model.eval()
    logits, _ = model(input_ids)
    return logits[0, -1].argmax().item()


@torch.no_grad()
def eval_bucket(model, tokenizer, queries, device):
    by_relation   = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total         = 0

    for q in queries:
        input_ids = tokenizer.encode(q.prompt, return_tensors="pt").to(device)

        pred_id = decode_answer(model, input_ids)
        gold_id = tokenizer.encode(q.answer)[0]

        correct = int(pred_id == gold_id)

        by_relation[q.relation]["correct"] += correct
        by_relation[q.relation]["total"]   += 1
        total_correct += correct
        total         += 1

    overall = total_correct / total if total > 0 else 0.0

    return {
        "overall": overall,
        "total":   total,
        "by_relation": {
            r: v["correct"] / v["total"]
            for r, v in by_relation.items() if v["total"] > 0
        },
    }
@torch.no_grad()
def print_examples(
    model,
    tokenizer,
    queries,
    device,
    title,
    max_examples=5,
):
    model.eval()

    print(f"\n{'═'*65}")
    print(f" EXAMPLES — {title}")
    print(f"{'═'*65}")

    shown = 0
    correct_count = 0

    for q in queries:
        if shown >= max_examples:
            break

        input_ids = tokenizer.encode(q.prompt, return_tensors="pt").to(device)
        pred_id   = decode_answer(model, input_ids)
        gold_id   = tokenizer.encode(q.answer)[0]

        pred_text = tokenizer.decode([pred_id])
        gold_text = tokenizer.decode([gold_id])

        correct = int(pred_id == gold_id)
        correct_count += correct
        status = "✓" if correct else "✗"

        print(f"\n{status} Prompt : {q.prompt}")
        print(f"   Gold   : {gold_text}")
        print(f"   Pred   : {pred_text}")
        print(f"   Rel    : {q.relation}")

        shown += 1

    print(f"\nAccuracy on these {shown}: {correct_count}/{shown}")

# ── Split test set into three sections ─────────────────────────
def split_queries(test_queries, held_out_names):
    extraction = [
        q for q in test_queries
        if q.query_type == "extraction"
    ]

    comp_seen = [
        q for q in test_queries
        if q.query_type == "composition"
        and q.subject.split()[0] not in held_out_names
    ]

    comp_heldout = [
        q for q in test_queries
        if q.query_type == "composition"
        and q.subject.split()[0] in held_out_names
    ]

    return extraction, comp_seen, comp_heldout


# ── Reporting ──────────────────────────────────────────────────
def print_report(name, section, relations, r):
    print(f"\n{'═'*55}")
    print(f" {name} — {section}")
    print(f"{'═'*55}")
    print(f"  Overall accuracy: {r['overall']:.4f}  ({r['total']} queries)")
    print(f"\n  By relation:")
    for rel in relations:
        acc = r["by_relation"].get(rel, 0)
        bar = "█" * int(acc * 30)
        print(f"    {rel:<14} {acc:.4f}  {bar}")


def print_comparison(section, relations, sr, cr):
    print(f"\n{'═'*65}")
    print(f" COMPARISON — {section}")
    print(f"{'═'*65}")
    print(f"  {'Relation':<14} {'Summed':>10} {'Disent':>10} {'Δ':>8}")
    print(f"  {'-'*50}")

    for rel in relations:
        s = sr["by_relation"].get(rel, 0)
        c = cr["by_relation"].get(rel, 0)
        arrow = "↑" if c > s else ("↓" if c < s else "=")
        print(f"  {rel:<14} {s:>10.4f} {c:>10.4f} {arrow} {c-s:>+.4f}")

    print(f"  {'-'*50}")
    print(
        f"  {'Overall':<14} "
        f"{sr['overall']:>10.4f} "
        f"{cr['overall']:>10.4f} "
        f"{'↑' if cr['overall'] > sr['overall'] else '↓'} "
        f"{cr['overall']-sr['overall']:>+.4f}"
    )




extraction, comp_seen, comp_heldout = split_queries(test_queries, held_out_names)
print_examples(
    summed_model,
    tokenizer,
    extraction,
    device,
    title="SUMMED — Extraction",
)

print_examples(
    summed_model,
    tokenizer,
    comp_seen,
    device,
    title="SUMMED — Composition (seen)",
)

print_examples(
    summed_model,
    tokenizer,
    comp_heldout,
    device,
    title="SUMMED — Composition (held-out)",
)
print_examples(
    disent_model,
    tokenizer,
    extraction,
    device,
    title="DISENTANGLED — Extraction",
)

print_examples(
    disent_model,
    tokenizer,
    comp_seen,
    device,
    title="DISENTANGLED — Composition (seen)",
)

print_examples(
    disent_model,
    tokenizer,
    comp_heldout,
    device,
    title="DISENTANGLED — Composition (held-out)",
)
print("Evaluating summed model...")
sum_ext  = eval_bucket(summed_model, tokenizer, extraction,  device)
sum_seen = eval_bucket(summed_model, tokenizer, comp_seen,   device)
sum_ho   = eval_bucket(summed_model, tokenizer, comp_heldout, device)

print("Evaluating disentangled model...")
dis_ext  = eval_bucket(disent_model, tokenizer, extraction,  device)
dis_seen = eval_bucket(disent_model, tokenizer, comp_seen,   device)
dis_ho   = eval_bucket(disent_model, tokenizer, comp_heldout, device)


# ── Print reports ──────────────────────────────────────────────
print_report("SUMMED", "Extraction", EXTRACTION_RELATIONS, sum_ext)
print_report("SUMMED", "Composition (seen)", COMPOSITION_RELATIONS, sum_seen)
print_report("SUMMED", "Composition (held-out)", COMPOSITION_RELATIONS, sum_ho)

print_report("DISENTANGLED", "Extraction", EXTRACTION_RELATIONS, dis_ext)
print_report("DISENTANGLED", "Composition (seen)", COMPOSITION_RELATIONS, dis_seen)
print_report("DISENTANGLED", "Composition (held-out)", COMPOSITION_RELATIONS, dis_ho)

print_comparison("Extraction", EXTRACTION_RELATIONS, sum_ext, dis_ext)
print_comparison("Composition (seen)", COMPOSITION_RELATIONS, sum_seen, dis_seen)
print_comparison("Composition (held-out)", COMPOSITION_RELATIONS, sum_ho, dis_ho)