"""
train.py — Model definitions and training loop.

Two embedding variants, everything else identical:
  Baseline:      x = token_emb(t) + pos_emb(p)              shared 768-dim
  Disentangled:  x = concat(token_emb(t)[640], pos_emb(p)[128])   protected subspaces

Architecture: 4 transformer blocks, 4 attention heads (as specified).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os


# Config

class Config:
    vocab_size    = 287
    max_seq_len   = 20
    d_model       = 128
    n_heads       = 4
    n_layers      = 4
    dropout       = 0
    lr            = 3e-4
    lr_min        = 5e-6
    lr_decay_steps = 1000000
    batch_size    = 2048
    grad_accum    = 1    
    max_steps     = 1100000
    warmup_steps  = 1000
    seed = 45
    patience = 10000
    eval_every = 400
    min_steps = 500000
    save_every = 10000



    # Disentangled split — must sum to d_model
    d_semantic    = 128
    d_positional  = 0


# Embeddings

class SummedEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(self, x):
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.drop(self.token_emb(x) + self.pos_emb(pos))


class DisentangledEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_semantic)
        self.drop = nn.Dropout(cfg.dropout)
        self.d_semantic = cfg.d_semantic
        self.d_positional = cfg.d_positional

        if cfg.d_positional > 0:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_positional)
        else:
            self.pos_emb = None

    def forward(self, x):
        B, T = x.shape
        sem = self.token_emb(x)

        if self.pos_emb is None:
            return self.drop(sem)

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        p = self.pos_emb(pos).expand(B, T, self.d_positional)
        return self.drop(torch.cat([sem, p], dim=-1))


# Transformer — 4 heads, 4 blocks

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim   = cfg.d_model,
            num_heads   = cfg.n_heads,
            batch_first = True,
        )
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_model * 4)
        self.fc2  = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        seq_len   = x.size(1)
        # Upper-triangular bool mask: True = ignore that position
        # so each token can only attend to itself and earlier tokens
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        attn_out, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask
        )
        x      = self.ln2(x + attn_out)
        ff_out = self.drop(self.fc2(F.gelu(self.fc1(x))))
        return x + ff_out


class TransformerLM(nn.Module):
    def __init__(self, cfg, embedding_type="summed"):
        super().__init__()
        self.embedding_type = embedding_type

        if embedding_type == "summed":
            self.embed = SummedEmbedding(cfg)
        else:
            self.embed = DisentangledEmbedding(cfg)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()
        print(f"  [{embedding_type}] {self.count_params():,} parameters  "
              f"| {cfg.n_layers} blocks | {cfg.n_heads} heads")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None, answer_mask = None):
        x      = self.embed(x)
        for block in self.blocks:
            x  = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
           # Only compute loss at answer token positions
            loss = F.cross_entropy(
                logits[answer_mask],
                targets[answer_mask],
                )
            
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_token_embedding(self, token_ids):
        """Return semantic subspace embedding for given token ids."""
        return self.embed.token_emb(token_ids)


# Training uti

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps

    progress = (step - cfg.warmup_steps) / max(
        1,
        cfg.lr_decay_steps - cfg.warmup_steps
    )
    progress = min(1.0, max(0.0, progress))

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


def train(embedding_type, cfg, train_dl, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    model     = TransformerLM(cfg, embedding_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    log        = {"steps": [], "train_loss": [], "val_loss": []}
    step       = 0
    train_iter = iter(train_dl)
    optimizer.zero_grad()

    # Early stopping on training loss
    best_loss         = float("inf")
    evals_without_imp = 0
    best_model_path   = f"{out_dir}/model_best.pt"

    print(f"  [{embedding_type}] early stopping: patience={cfg.patience}, "
          f"eval_every={cfg.eval_every}")

    while step < cfg.max_steps:
        model.train()
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y, mask = next(train_iter)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            _, loss = model(x, y, answer_mask = mask)
            (loss / cfg.grad_accum).backward()
            accum += loss.item() / cfg.grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
                   
        if step % 200 == 0:
            print(f"  [{embedding_type}] step {step:>6} | loss {accum:.6f} | lr {lr:.2e}")
        
        if step % cfg.eval_every == 0:
            log["steps"].append(step)
            log["train_loss"].append(round(accum, 6))

            if accum < best_loss:
                
                best_loss = accum
                evals_without_imp = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  [{embedding_type}] step {step:>6} | loss {accum:.6f}  ✓ best")
            else:
                evals_without_imp += 1
                print(f"  [{embedding_type}] step {step:>6} | loss {accum:.6f}"
                      f"  (no improvement {evals_without_imp}/{cfg.patience})")

            if step >= cfg.min_steps and evals_without_imp >= cfg.patience:
                print(f"  [{embedding_type}] early stopping at step {step}. "
                      f"Best loss {best_loss:.6f}")
                break

    # Restore best weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"  [{embedding_type}] restored best weights")

    torch.save(model.state_dict(), f"{out_dir}/model_final.pt")
    log["best_loss"] = round(best_loss, 6)
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [{embedding_type}] training complete → {out_dir}/model_final.pt")
    return model

def load_model(path, embedding_type, cfg, device):
    model = TransformerLM(cfg, embedding_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


import torch
import argparse
from dataset   import build_dataset, save_dataset, get_dataloaders, load_dataset, build_capacity_dataset
from tokenizer import SROTokenizer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["summed", "disentangled", "both"], default="both")
    parser.add_argument("--out",    default="./outputs")
    parser.add_argument("--steps",  type=int, default=None)
    parser.add_argument("--reload", action="store_true",
                        help="Reload existing dataset.json instead of regenerating")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    cfg = Config()
    tokenizer = SROTokenizer.load("tokenizer10000.json")
    cfg.vocab_size = tokenizer.vocab_size
    entities, entity_index, train_queries, test_queries, held_out = load_dataset(
        "dataset_extraction10000.json"
    )
   

    # Dataloaders
    print("\nBuilding dataloaders...")
    train_dl = get_dataloaders(
    train_queries,
    tokenizer,
    batch_size=cfg.batch_size,
    seed=cfg.seed,
)    # Train
   
    if args.model in ("disentangled", "both"):
        print(f"\n{'='*55}")
        print("Training: DISENTANGLED (method)")
        print(f"{'='*55}")
        train("disentangled", cfg, train_dl, device, out_dir=f"{args.out}/disentangled")

    print("\nTraining complete.")


import torch
import torch.nn.functional as F
from tokenizer import SROTokenizer
from dataset import load_dataset, EXTRACTION_RELATIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg    = Config()

entities, entity_index, train_queries, test_queries, held_out_names = load_dataset("dataset_extraction10000.json")

tokenizer      = SROTokenizer.from_dataset("dataset_extraction10000.json")
cfg.vocab_size = tokenizer.vocab_size


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






print("Evaluating disentangled model...")
disent_results = eval_extraction(disent_model, tokenizer, test_queries, device)

print_report("DISENTANGLED (method)", disent_results)

