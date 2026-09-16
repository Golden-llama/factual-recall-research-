import argparse

import torch

from dataset_composition import get_dataloaders, load_dataset
from tokenizer_composition import SROTokenizer
from train_composition import Config, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["summed", "disentangled", "both"], default="both")
    parser.add_argument("--out", default="./outputs")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Kept for compatibility; this script loads existing dataset/tokenizer files.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every

    tokenizer = SROTokenizer.load("tokenizer1000.json")
    cfg.vocab_size = tokenizer.vocab_size

    entities, train_queries, val_queries, test_queries = load_dataset(
        "dataset_composition1000.json"
    )

    print("\nBuilding dataloaders...")
    train_dl = get_dataloaders(
        train_queries,
        tokenizer,
        batch_size=cfg.batch_size, seed = cfg.seed
    )
    unseen_dl = get_dataloaders(
        val_queries,
        tokenizer,
        batch_size=cfg.batch_size, seed = cfg.seed
    )


    if args.model in ("disentangled", "both"):
        print(f"\n{'=' * 55}")
        print("Training: DISENTANGLED (method)")
        print(f"{'=' * 55}")
        train(
            "disentangled",
            cfg,
            train_dl,
            device,
            out_dir=f"{args.out}/disentangled",
            unseen_dl=unseen_dl,
            train_eval_dl=train_dl,
        )

    print("\nTraining complete.")