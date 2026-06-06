"""
train_experiment.py — Wires dataset + tokenizer + training loop together.

Usage:
    python train_experiment.py --model both --out ./outputs
    python train_experiment.py --model both --out ./outputs --steps 2000   # quick pilot
"""

import torch
import argparse
from train     import Config, train, load_model
from dataset   import build_dataset, save_dataset, get_dataloaders, load_dataset
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

    # Dataset
    entities, entity_index, train_queries, test_queries, held_out_names = load_dataset("dataset.json")
    
    tokenizer = SROTokenizer.load("tokenizer.json")


    # Config
    cfg = Config()
    cfg.vocab_size = len(tokenizer)
    if args.steps:
        cfg.max_steps = args.steps

    # Dataloaders
    print("\nBuilding dataloaders...")
    train_dl, val_dl = get_dataloaders(train_queries, tokenizer, cfg, batch_size=cfg.batch_size)

    # Train
    if args.model in ("summed", "both"):
        print(f"\n{'='*55}")
        print("Training: SUMMED (baseline)")
        print(f"{'='*55}")
        train("summed", cfg, train_dl, val_dl, device, out_dir=f"{args.out}/summed")

    if args.model in ("disentangled", "both"):
        print(f"\n{'='*55}")
        print("Training: DISENTANGLED (method)")
        print(f"{'='*55}")
        train("disentangled", cfg, train_dl, val_dl, device, out_dir=f"{args.out}/disentangled")

    print("\nTraining complete.")