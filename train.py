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
import argparse


# Config

class Config:
    vocab_size    = 1052       
    max_seq_len   = 128
    d_model       = 256
    n_heads       = 4          # as specified
    n_layers      = 4          # as specified
    dropout       = 0.1
    lr            = 1e-4
    batch_size    = 32
    grad_accum    = 2          # effective batch = 64
    max_steps     = 30000   # hard ceiling — early stopping will trigger first
    warmup_steps  = 500
    eval_every    = 200       # evaluate every N steps
    patience      = 5         # stop if val_ppl doesn't improve for this many evals
    min_steps     = 1000     # don't stop before this many steps (let model warm up)
    seed          = 42

    # Disentangled split — must sum to d_model
    d_semantic    = 192
    d_positional  = 64


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
    """
    Experimental: semantic subspace (640-dim) and positional subspace (128-dim)
    concatenated — gradients cannot cross subspaces.
    Total dim = 768, matched to baseline.
    """
    def __init__(self, cfg):
        super().__init__()
        self.token_emb   = nn.Embedding(cfg.vocab_size,  cfg.d_semantic)
        self.pos_emb     = nn.Embedding(cfg.max_seq_len, cfg.d_positional)
        self.drop        = nn.Dropout(cfg.dropout)
        self.d_semantic  = cfg.d_semantic
        self.d_positional = cfg.d_positional

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        sem  = self.token_emb(x)                                        # (B, T, 640)
        p    = self.pos_emb(pos).expand(B, T, self.d_positional)        # (B, T, 128)
        return self.drop(torch.cat([sem, p], dim=-1))                   # (B, T, 768)


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

    def forward(self, x, targets=None):
        x      = self.embed(x)
        for block in self.blocks:
            x  = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0)
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
    # Cosine decay over max_steps — early stopping triggers before hitting this ceiling
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_ppl(model, val_dl, device, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(val_dl):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    return math.exp(total_loss / n) if n else float("inf")


def train(embedding_type, cfg, train_dl, val_dl, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    model     = TransformerLM(cfg, embedding_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    log        = {"steps": [], "train_loss": [], "val_ppl": []}
    step       = 0
    train_iter = iter(train_dl)
    optimizer.zero_grad()

    # Early stopping state
    best_val_ppl      = float("inf")
    best_step         = 0
    evals_without_imp = 0   # counts consecutive evals with no improvement
    best_model_path   = f"{out_dir}/model_best.pt"

    print(f"  [{embedding_type}] early stopping: patience={cfg.patience}, "
          f"eval_every={cfg.eval_every}, min_steps={cfg.min_steps}")

    while step < cfg.max_steps:
        model.train()
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            accum += loss.item() / cfg.grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 200 == 0:
            print(f"  [{embedding_type}] step {step:>6} | loss {accum:.4f} | lr {lr:.2e}")

        if step % cfg.eval_every == 0:
            ppl = evaluate_ppl(model, val_dl, device)
            log["steps"].append(step)
            log["train_loss"].append(round(accum, 4))
            log["val_ppl"].append(round(ppl, 4))

            if ppl < best_val_ppl:
                # Improvement — save best model and reset patience counter
                best_val_ppl      = ppl
                best_step         = step
                evals_without_imp = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  [{embedding_type}] step {step:>6} | val_ppl {ppl:.4f}  ✓ best")
            else:
                evals_without_imp += 1
                print(f"  [{embedding_type}] step {step:>6} | val_ppl {ppl:.4f}"
                      f"  (no improvement {evals_without_imp}/{cfg.patience})")

            # Early stopping check — only after min_steps
            if step >= cfg.min_steps and evals_without_imp >= cfg.patience:
                print(f"  [{embedding_type}] early stopping at step {step}. "
                      f"Best val_ppl {best_val_ppl:.4f} at step {best_step}.")
                break

    # Load best weights before saving final model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"  [{embedding_type}] restored best weights from step {best_step}")

    torch.save(model.state_dict(), f"{out_dir}/model_final.pt")
    log["best_step"]    = best_step
    log["best_val_ppl"] = round(best_val_ppl, 4)
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [{embedding_type}] training complete → {out_dir}/model_final.pt")
    return model


def load_model(path, embedding_type, cfg, device):
    model = TransformerLM(cfg, embedding_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model