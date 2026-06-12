import torch
import torch.nn.functional as F
from collections import defaultdict
from tokenizer import SROTokenizer
from train import Config, load_model
from dataset import load_dataset, EXTRACTION_RELATIONS, COMPOSITION_RELATIONS

# ── Load ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg    = Config()

entities, entity_index, train_queries, val_queries, test_queries, held_out_names = \
    load_dataset("dataset_composition.json")

tokenizer      = SROTokenizer.load("tokenizer.json")
cfg.vocab_size = tokenizer.vocab_size
cfg.d_model      = 256
cfg.d_semantic   = 192
cfg.d_positional = 64
cfg.max_seq_len  = 20

summed_model = load_model("outputs/summed/model_best.pt",       "summed",       cfg, device)
disent_model = load_model("outputs/disentangled/model_best.pt", "disentangled", cfg, device)


# ═══════════════════════════════════════════════════════════════
# Core scoring
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def score_query(model, tokenizer, query, device):
    input_ids = tokenizer.encode(query.prompt, return_tensors="pt").to(device)
    logits, _ = model(input_ids)
    probs     = F.softmax(logits[0, -1, :], dim=-1)
    ans_tok   = tokenizer.encode(query.answer)[0]
    correct   = int(probs.argmax().item() == ans_tok)
    top3_ids  = probs.argsort(descending=True)[:3].tolist()
    top3      = [(tokenizer.convert_ids_to_tokens(i), round(probs[i].item(), 3))
                 for i in top3_ids]
    return correct, top3


# ═══════════════════════════════════════════════════════════════
# Evaluation — three buckets
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, tokenizer, test_queries, held_out_names, device):
    model.eval()

    extraction   = [q for q in test_queries if q.query_type == "extraction"]
    comp_seen    = [q for q in test_queries
                    if q.query_type == "composition"
                    and q.subject.split()[0] not in held_out_names]
    comp_heldout = [q for q in test_queries
                    if q.query_type == "composition"
                    and q.subject.split()[0] in held_out_names]

    def acc(queries):
        if not queries:
            return 0.0, 0
        correct = sum(score_query(model, tokenizer, q, device)[0] for q in queries)
        return correct / len(queries), len(queries)

    return {
        "extraction":          acc(extraction),
        "composition_seen":    acc(comp_seen),
        "composition_heldout": acc(comp_heldout),
    }


# ═══════════════════════════════════════════════════════════════
# Print examples
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def print_examples(model, model_name, test_queries, held_out_names, device, n=5):
    model.eval()
    print(f"\n{'═'*65}")
    print(f" EXAMPLES — {model_name}")
    print(f"{'═'*65}")

    buckets = {
        "Extraction (all entities)":         [q for q in test_queries
                                              if q.query_type == "extraction"][:n],
        "Composition — seen entities":       [q for q in test_queries
                                              if q.query_type == "composition"
                                              and q.subject.split()[0]
                                              not in held_out_names][:n],
        "Composition — held-out (gen)":      [q for q in test_queries
                                              if q.query_type == "composition"
                                              and q.subject.split()[0]
                                              in held_out_names][:n],
    }

    for bucket_name, queries in buckets.items():
        print(f"\n── {bucket_name} ──")
        correct_count = 0
        for q in queries:
            correct, top3 = score_query(model, tokenizer, q, device)
            correct_count += correct
            status = "✓" if correct else "✗"
            print(f"  {status}  Input:    {q.prompt}")
            print(f"      Expected: {q.answer}")
            print(f"      Top 3:    {top3[0][0]}({top3[0][1]})  "
                  f"{top3[1][0]}({top3[1][1]})  {top3[2][0]}({top3[2][1]})")
        print(f"  Accuracy on these {n}: {correct_count}/{n}")


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def print_report(name, r):
    print(f"\n{'═'*55}")
    print(f" {name}")
    print(f"{'═'*55}")
    labels = {
        "extraction":          "Extraction — all entities    ",
        "composition_seen":    "Composition — seen (train)   ",
        "composition_heldout": "Composition — held-out (gen) ",
    }
    for key, label in labels.items():
        acc, n = r[key]
        bar    = "█" * int(acc * 30)
        print(f"  {label}  {acc:.4f}  {bar}  (n={n})")


def print_comparison(sr, cr):
    print(f"\n{'═'*65}")
    print(" COMPARISON: Summed (baseline) vs Disentangled (method)")
    print(f"{'═'*65}")
    labels = {
        "extraction":          "Extraction — all entities    ",
        "composition_seen":    "Composition — seen (train)   ",
        "composition_heldout": "Composition — held-out (gen) ",
    }
    for key, label in labels.items():
        s, sn = sr[key]
        c, cn = cr[key]
        delta  = c - s
        arrow  = "↑ method" if delta > 0 else ("↓ method" if delta < 0 else "=")
        print(f"  {label}  summed={s:.4f}  disent={c:.4f}  "
              f"{arrow} ({delta:+.4f})  (n={sn})")


# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════

print("Evaluating summed model...")
summed_results = evaluate(summed_model, tokenizer, test_queries, held_out_names, device)
print_report("SUMMED (baseline)", summed_results)
print_examples(summed_model, "SUMMED", test_queries, held_out_names, device)

print("\nEvaluating disentangled model...")
disent_results = evaluate(disent_model, tokenizer, test_queries, held_out_names, device)
print_report("DISENTANGLED (method)", disent_results)
print_examples(disent_model, "DISENTANGLED", test_queries, held_out_names, device)

print_comparison(summed_results, disent_results)