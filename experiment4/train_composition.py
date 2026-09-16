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
    vocab_size    = 1053
    max_seq_len   = 20
    d_model       = 128
    n_heads       = 4          # as specified
    n_layers      = 4          # as specified
    dropout       = 0
    lr            = 3e-4
    weight_decay = 0.01
    batch_size    = 128
    grad_accum    = 1          
    max_steps     = 150000   # hard ceiling — early stopping will trigger first
    warmup_steps  = 1000
    min_steps     = 1000     # don't stop before this many steps (let model warm up)
    seed          = 34
    save_every = 10000
    eval_every = 400


    # Disentangled split — must sum to d_model
    d_semantic    = 128
    d_positional  = 0


# Embeddings

class SummedEmbedding(nn.Module):
    """Baseline: token + positional embeddings summed into shared 768-dim space."""
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


# Training utilities

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    return cfg.lr

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y, mask in dataloader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        preds = logits[mask].argmax(dim=-1)
        targets = y[mask]
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return correct / total if total else 0.0

def train(embedding_type, cfg, train_dl, device, out_dir, unseen_dl=None, train_eval_dl=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = TransformerLM(cfg, embedding_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    log = {
        "steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "unseen_composition_accuracy": [],
    }
    step = 0
    train_iter = iter(train_dl)
    optimizer.zero_grad()

    best_unseen_acc = -1.0
    best_unseen_step = None
    best_model_path = f"{out_dir}/model_best.pt"

    print(
        f"  [{embedding_type}] grokking run: max_steps={cfg.max_steps}, "
        f"eval_every={cfg.eval_every}, weight_decay={cfg.weight_decay:g}, seed={cfg.seed}"
    )
    print(f"  [{embedding_type}] no early stopping; checkpointing best held-out accuracy")

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
            _, loss = model(x, y, answer_mask=mask)
            (loss / cfg.grad_accum).backward()
            accum += loss.item() / cfg.grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 200 == 0:
            print(f"  [{embedding_type}] step {step:>6} | loss {accum:.6f} | lr {lr:.2e}")

        if step % cfg.eval_every == 0:
            train_acc = evaluate_accuracy(model, train_eval_dl, device) if train_eval_dl is not None else None
            unseen_acc = evaluate_accuracy(model, unseen_dl, device) if unseen_dl is not None else None
            log["steps"].append(step)
            log["train_loss"].append(round(accum, 6))
            log["train_accuracy"].append(round(train_acc, 6) if train_acc is not None else None)
            log["unseen_composition_accuracy"].append(
                round(unseen_acc, 6) if unseen_acc is not None else None
            )

            parts = [f"  [{embedding_type}] eval step {step:>6}", f"loss {accum:.6f}"]
            if train_acc is not None:
                parts.append(f"train acc {train_acc:.4f}")
            if unseen_acc is not None:
                parts.append(f"held-out comp acc {unseen_acc:.4f}")
                if unseen_acc > best_unseen_acc:
                    best_unseen_acc = unseen_acc
                    best_unseen_step = step
                    torch.save(model.state_dict(), best_model_path)
                    parts.append("best-heldout")
            print(" | ".join(parts))

        if cfg.save_every and step % cfg.save_every == 0:
            torch.save(model.state_dict(), f"{out_dir}/model_step_{step}.pt")

    if best_unseen_step is not None:
        print(
            f"  [{embedding_type}] best held-out composition accuracy "
            f"{best_unseen_acc:.4f} at step {best_unseen_step}"
        )
    else:
        torch.save(model.state_dict(), best_model_path)
        print(f"  [{embedding_type}] no held-out eval; saved final weights as best checkpoint")

    torch.save(model.state_dict(), f"{out_dir}/model_final.pt")
    log["best_unseen_composition_accuracy"] = (
        round(best_unseen_acc, 6) if best_unseen_step is not None else None
    )
    log["best_unseen_step"] = best_unseen_step
    log["weight_decay"] = cfg.weight_decay
    log["seed"] = cfg.seed
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [{embedding_type}] training complete -> {out_dir}/model_final.pt")
    return model


def load_model(path, embedding_type, cfg, device):
    model = TransformerLM(cfg, embedding_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model