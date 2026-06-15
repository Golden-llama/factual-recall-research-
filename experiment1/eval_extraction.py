import torch
import torch.nn.functional as F
from tokenizer import SROTokenizer
from train_extraction import Config, load_model
from dataset_extraction import load_dataset, EXTRACTION_RELATIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg    = Config()

entities, entity_index, train_queries, test_queries, held_out_names = load_dataset("dataset_extraction1000.json")

tokenizer      = SROTokenizer.from_dataset("dataset_extraction1000.json")
cfg.vocab_size = tokenizer.vocab_size
cfg.d_model    = 120
cfg.d_semantic    = 90
cfg.d_positional  = 30
cfg.max_seq_len   = 20

summed_model = load_model("outputs/summed/model_best.pt",       "summed",       cfg, device)
disent_model = load_model("outputs/disentangled/model_best.pt", "disentangled", cfg, device)
@torch.no_grad()
def decode_answer(model, tokenizer, input_ids):
    
    model.eval()
    out = input_ids.clone()

   
    logits, _ = model(out)
    next_tok = logits[0, -1].argmax().item()
        

    return next_tok

@torch.no_grad()
def eval_extraction(
    model,
    tokenizer,
    test_queries,
    device,
    print_examples=True,
    max_examples=10
):
    model.eval()

    from collections import defaultdict
    by_relation   = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total         = 0

    printed = 0

    for q in test_queries:
        if q.query_type != "extraction":
            continue

        input_ids = tokenizer.encode(q.prompt, return_tensors="pt").to(device)

        # Decode prediction
        pred_ids = decode_answer(model, tokenizer, input_ids)
        pred_text = tokenizer.decode([pred_ids])

        gold_ids  = tokenizer.encode(q.answer)[0]
        gold_text = tokenizer.decode([gold_ids])

        correct = int(pred_ids == gold_ids)

        by_relation[q.relation]["correct"] += correct
        by_relation[q.relation]["total"]   += 1
        total_correct += correct
        total         += 1

        # ---- PRINT EXAMPLES ----
        if print_examples and printed < max_examples:
            print("\n" + "─" * 60)
            print(f"Subject : {q.subject}")
            print(f"Relation: {q.relation}")
            print(f"Prompt  : {q.prompt}")
            print(f"Gold    : {gold_text}")
            print(f"Pred    : {pred_text}")
            print(f"Correct : {'✓' if correct else '✗'}")
            printed += 1

    overall = total_correct / total

    return {
        "overall":     overall,
        "total":       total,
        "by_relation": {
            rel: v["correct"] / v["total"]
            for rel, v in by_relation.items() if v["total"] > 0
        },
    }


def print_report(name, r):
    print(f"\n{'═'*55}")
    print(f" {name}")
    print(f"{'═'*55}")
    print(f"  Overall extraction accuracy: {r['overall']:.4f}  ({r['total']} queries)")
    print(f"\n  By relation:")
    for rel in EXTRACTION_RELATIONS:
        acc = r["by_relation"].get(rel, 0)
        bar = "█" * int(acc * 30)
        print(f"    {rel:<12} {acc:.4f}  {bar}")


def print_comparison(sr, cr):
    print(f"\n{'═'*55}")
    print(" COMPARISON")
    print(f"{'═'*55}")
    print(f"  {'Relation':<14} {'Summed':>10} {'Disentangled':>14} {'Δ':>8}")
    print(f"  {'-'*50}")
    for rel in EXTRACTION_RELATIONS:
        s = sr["by_relation"].get(rel, 0)
        c = cr["by_relation"].get(rel, 0)
        arrow = "↑" if c > s else ("↓" if c < s else "=")
        print(f"  {rel:<14} {s:>10.4f} {c:>14.4f} {arrow} {c-s:>+.4f}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':<14} {sr['overall']:>10.4f} {cr['overall']:>14.4f} "
          f"{'↑' if cr['overall'] > sr['overall'] else '↓'} "
          f"{cr['overall']-sr['overall']:>+.4f}")


print("Evaluating summed model...")
summed_results = eval_extraction(summed_model, tokenizer, test_queries, device)

print("Evaluating disentangled model...")
disent_results = eval_extraction(disent_model, tokenizer, test_queries, device)

print_report("SUMMED (baseline)",     summed_results)
print_report("DISENTANGLED (method)", disent_results)
print_comparison(summed_results, disent_results)

