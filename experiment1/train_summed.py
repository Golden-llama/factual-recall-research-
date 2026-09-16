"""
train_experiment.py — Wires dataset + tokenizer + training loop together.

Usage:
    python train_experiment.py --model both --out ./outputs
    python train_experiment.py --model both --out ./outputs --steps 2000   # quick pilot
"""

import torch
import argparse
from train_extraction     import Config, train, load_model
from dataset_extraction   import build_dataset, save_dataset, get_dataloaders, load_dataset, build_capacity_dataset
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
)
    # Train
    if args.model in ("summed", "both"):
        print(f"\n{'='*55}")
        print("Training: SUMMED (baseline)")
        print(f"{'='*55}")
        train("summed", cfg, train_dl, device, out_dir=f"{args.out}/summed")
    
    
    print("\nTraining complete.")


    

