import os
import torch
from torch.utils.data import DataLoader

from utils.probes import run_probe_collection
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from checkpoint if available, else use config/cfg_pretrain.yaml
    # Minimal example assumes pretrain config dict available via torch.load or YAML
    # Here we create a small default config; adjust as needed.
    config = {
        "batch_size": 32,
        "seq_len": 128,
        "puzzle_emb_ndim": 64,
        "num_puzzle_identifiers": 1024,
        "vocab_size": 512,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 2.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 8,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "bfloat16",
    }

    # Instantiate model
    model = HierarchicalReasoningModel_ACTV1(config)
    model.to(device)

    # Load dataset (adjust path to your dataset)
    dataset_path = os.path.join("data", "sudoku-extreme-1k-aug-1000", "test", "dataset.json")
    ds = PuzzleDataset(dataset_path)
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=False)

    # Run probe collection
    output_dir = os.path.join("results", "probes")
    run_probe_collection(model, dl, device=device, output_dir=output_dir)
    print(f"Saved probe datasets to {output_dir}")


if __name__ == "__main__":
    main()
