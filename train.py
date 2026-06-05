import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import argparse
 
class Config:
    vocab_size    = 50264      # GPT-2 50257 + 6 special tokens; updated after tokenizer extension
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
        pos = torch.arange(512, device = x.device).unsqueeze(0)
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
        pos = torch.arange(512, device = x.device).unsqueeze(0)
        sem = self.token_emb(x)
        p = self.pos_emb(pos).expand(B,T, self.d_positional)
        return self.drop(torch.cat([sem,p], dim = -1))
    
class TransformerBlock(nn.Module):
    def __init__(self.cfg):
        super().__init__()
        




 
 