import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import argparse
 
class Config:
    vocab_size    = 1052      
    max_seq_len   = 512
    d_model       = 768
    n_heads       = 4          # as specified
    n_layers      = 4          # as specified
    ffn_mult      = 4
    dropout       = 0.1
    lr            = 3e-4
    batch_size    = 32
    grad_accum    = 2          # effective batch = 64
    max_steps     = 30_000
    warmup_steps  = 1_000
    eval_every    = 1_000
    save_every    = 10_000
    seed          = 42
 
    # Disentangled split 
    d_semantic    = 640
    d_positional  = 128
#for added embedding

class SummedEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device = x.device).unsqueeze(0)
        return self.drop(self.token_emb(x)+self.pos_emb(pos))
#for concatenated embeddings   

class DisentangledEmbedding(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_semantic)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_positional)
        self.drop = nn.Dropout(cfg.dropout)
        self.d_semantic  = cfg.d_semantic
        self.d_positional = cfg.d_positional
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device = x.device).unsqueeze(0)
        sem = self.token_emb(x)
        p = self.pos_emb(pos).expand(B,T, self.d_positional)
        return self.drop(torch.cat([sem,p], dim = -1))
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=768,
            num_heads=4,
            batch_first=True
        )
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_model*4)
        self.fc2 = nn.Linear(cfg.d_model*4, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len,seq_len, device = x.device), diagonal = 1).bool()
        attn_out, __ = self.attn(self.ln1(x),self.ln1(x),self.ln1(x), attn_mask = attn_mask)
        x = self.ln2(x+attn_out)
        ff_out = self.drop(self.fc2(F.gelu(self.fc1(x))))
        return (x + ff_out)


class TransformerLM(nn.Module):
    def __init__(self, cfg, embedding_type = "summed"):
        super().__init__()
        self.embedding_type = embedding_type
        if embedding_type == "summed":
            self.embed = SummedEmbedding(cfg)
        else:
            self.embed = DisentangledEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(4)])
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias = False)
        self.ln = nn.LayerNorm(cfg.d_model)
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
    def forward(self,x, targets = None):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.unembed(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_token_embedding(self, token_ids):
        """Return semantic subspace embedding for given token ids."""
        return self.embed.token_emb(token_ids)
    

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
@torch.no_grad()
def evaluate_ppl(model, val_dl, device, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y, _) in enumerate(val_dl):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    return math.exp(total_loss / n) if n else float("inf")

def train(embedding_type, cfg, train_dl, val_dl, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
 
    model = TransformerLM(cfg, embedding_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
 
    log        = {"steps": [], "train_loss": [], "val_ppl": []}
    step       = 0
    train_iter = iter(train_dl)
    optimizer.zero_grad()
 
    while step < cfg.max_steps:
        model.train()
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
 
        accum = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y, _ = next(train_iter)
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
            print(f"  [{embedding_type}] step {step:>6} | val_ppl {ppl:.2f}")
 
        if step % cfg.save_every == 0:
            torch.save(model.state_dict(), f"{out_dir}/ckpt_step{step}.pt")
 
    torch.save(model.state_dict(), f"{out_dir}/model_final.pt")
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [{embedding_type}] training complete → {out_dir}/model_final.pt")
    return model

def load_model(path, embedding_type, cfg, device):
    model = TransformerLM(cfg, embedding_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
 
    


        






 
 