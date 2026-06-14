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
    vocab_size    = 1053      
    max_seq_len   = 20
    d_model       = 256
    n_heads       = 4          # as specified
    n_layers      = 4          # as specified
    dropout       = 0
    lr            = 1e-3
    batch_size    = 32
    grad_accum    = 2          # effective batch = 64
    max_steps     = 100000   # hard ceiling — early stopping will trigger first
    warmup_steps  = 200
    min_steps     = 1000     # don't stop before this many steps (let model warm up)
    seed          = 42
    save_every = 10000
    eval_every = 500
    patience = 20


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

    def forward(self, x, targets=None, answer_mask=None):
        x      = self.embed(x)
        for block in self.blocks:
            x  = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            if answer_mask is not None:
            # Only compute loss at answer token positions
                loss = F.cross_entropy(
                    logits[answer_mask],
                    targets[answer_mask],
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0
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
    # Cosine decay over max_steps — early stopping triggers before hitting this ceiling
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate_val(model, val_dl, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            n += 1
    return total_loss / n if n else float("inf")

def train(embedding_type, cfg, train_dl, val_dl, device, out_dir):
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
        '''
        if step % cfg.eval_every == 0:
            log["steps"].append(step)
            log["train_loss"].append(round(accum, 6))
            val_loss = evaluate_val(model, val_dl, device)
            log["val_loss"].append(round(val_loss, 6))

            if val_loss < best_loss:
                best_loss         = val_loss
                evals_without_imp = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  [{embedding_type}] step {step:>6} | "
                    f"val_loss {val_loss:.6f}  ✓ best")
            else:
                evals_without_imp += 1
                print(f"  [{embedding_type}] step {step:>6} | "
                  f"val_loss {val_loss:.6f}  "
                  f"(no improvement {evals_without_imp}/{cfg.patience})")

            if step >= cfg.min_steps and evals_without_imp >= cfg.patience:
                print(f"  [{embedding_type}] early stopping at step {step}")
                break
        '''
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