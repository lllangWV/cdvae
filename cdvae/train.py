#!/usr/bin/env python3
"""
A standalone training script without Hydra or PyTorch Lightning.
All configuration/hyperparameter settings are defined at the top.
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import DataLoader

# (Assuming these exist in your codebase)
from cdvae.common.data_utils import get_scaler_from_data_list
from cdvae.common.utils import PROJECT_ROOT, log_hyperparameters
from cdvae.data.dataset import CrystDataset
from cdvae.models.cdvae_model import CDVAE, hparams

###############################################################################
# CONFIGURATION
###############################################################################
# All hyperparameters and configuration options are stored in the CONFIG dict.
CONFIG = OmegaConf.create(
    {
        "data": {
            "root_path": os.path.join(PROJECT_ROOT, "data/mp_20"),
            "prop": "formation_energy_per_atom",
            "num_targets": 1,
            "niggli": True,
            "primitive": False,
            "graph_method": "crystalnn",
            "lattice_scale_method": "scale_length",
            "preprocess_workers": 30,
            "readout": "mean",
            "max_atoms": 20,
            "otf_graph": False,
            "eval_model_name": "mp20",
            "num_workers": {"train": 0, "val": 0, "test": 0},
            "batch_size": {"train": 256, "val": 256, "test": 256},
            "train": {
                "path": "data/mp_20/train.csv",
                "prop": "formation_energy_per_atom",
                "niggli": True,
                "primitive": False,
                "graph_method": "crystalnn",
                "lattice_scale_method": "scale_length",
                "preprocess_workers": 30,
            },
            "val": {
                "path": "data/mp_20/val.csv",
                "prop": "formation_energy_per_atom",
                "niggli": True,
                "primitive": False,
                "graph_method": "crystalnn",
                "lattice_scale_method": "scale_length",
                "preprocess_workers": 30,
            },
            "test": {
                "path": "data/mp_20/test.csv",
                "prop": "formation_energy_per_atom",
                "niggli": True,
                "primitive": False,
                "graph_method": "crystalnn",
                "lattice_scale_method": "scale_length",
                "preprocess_workers": 30,
            },
        },
        "train": {
            "deterministic": True,
            "random_seed": 42,
            "fast_dev_run": False,
            "num_epochs": 1000,
            "monitor_metric": "val_loss",
            "monitor_metric_mode": "min",
            "early_stopping": {
                "patience": 100000,
                "verbose": True,
            },
            "model_checkpoints": {
                "save_top_k": 1,
                "verbose": True,
            },
            "teacher_forcing_max_epoch": 500,
        },
        "logging": {
            "val_check_interval": 1,
            "progress_bar_refresh_rate": 20,
        },
        "optim": {
            "optimizer_params": {"lr": 1e-3},
        },
        "core": {
            "tags": ["experiment", "test"],
        },
    }
)


config = {}

###############################################################################


###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch: int, path: Path):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, str(path))
    print(f"Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, path: Path) -> int:
    checkpoint = torch.load(str(path))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint: {path}")
    return checkpoint["epoch"]


###############################################################################
# MAIN FUNCTION
###############################################################################
# def main():
# Determine device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set deterministic seed if required.
if CONFIG["train"]["deterministic"]:
    set_seed(CONFIG["train"]["random_seed"])

# Create a run directory (you could also mimic PROJECT_ROOT/conf settings)
run_dir = Path("runs") / "my_experiment"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {run_dir.resolve()}")

# Instantiate datamodule.
# dm_cfg = CONFIG["data"]["datamodule"]
# datamodule = MyDataModule(
#     batch_size=dm_cfg["batch_size"],
#     num_workers=dm_cfg["num_workers"],
# )

train_dataset = CrystDataset(
    name="train",
    path=CONFIG.data.train.path,
    prop=CONFIG.data.train.prop,
    niggli=CONFIG.data.train.niggli,
    primitive=CONFIG.data.train.primitive,
    graph_method=CONFIG.data.train.graph_method,
    lattice_scale_method=CONFIG.data.train.lattice_scale_method,
    preprocess_workers=CONFIG.data.train.preprocess_workers,
)

test_dataset = CrystDataset(
    name="test",
    path=CONFIG.data.test.path,
    prop=CONFIG.data.test.prop,
    niggli=CONFIG.data.test.niggli,
    primitive=CONFIG.data.test.primitive,
    graph_method=CONFIG.data.test.graph_method,
    lattice_scale_method=CONFIG.data.test.lattice_scale_method,
    preprocess_workers=CONFIG.data.test.preprocess_workers,
)


val_dataset = CrystDataset(
    name="val",
    path=CONFIG.data.val.path,
    prop=CONFIG.data.val.prop,
    niggli=CONFIG.data.val.niggli,
    primitive=CONFIG.data.val.primitive,
    graph_method=CONFIG.data.val.graph_method,
    lattice_scale_method=CONFIG.data.val.lattice_scale_method,
    preprocess_workers=CONFIG.data.val.preprocess_workers,
)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=CONFIG.data.batch_size.train,
    num_workers=CONFIG.data.num_workers.train,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=CONFIG.data.batch_size.test,
    num_workers=CONFIG.data.num_workers.test,
)


val_dataloader = DataLoader(
    val_dataset,
    batch_size=CONFIG.data.batch_size.val,
    num_workers=CONFIG.data.num_workers.val,
)


# Instantiate model.
model = CDVAE(hparams=hparams)


# (Pass scalers from datamodule to model, if applicable.)

# Here we assume both are dicts; in your code these could be objects with a copy() method.
model.lattice_scaler = get_scaler_from_data_list(
    train_dataset.cached_data, key="scaled_lattice"
)
model.scaler = get_scaler_from_data_list(
    train_dataset.cached_data, key=train_dataset.prop
)

# Save scalers.
torch.save(model.lattice_scaler.lattice_scaler, run_dir / "lattice_scaler.pt")
torch.save(model.scaler.scaler, run_dir / "prop_scaler.pt")

# Instantiate optimizer.
optim_cfg = CONFIG["optim"]
optimizer = torch.optim.Adam(model.parameters(), **optim_cfg["optimizer_params"])

# Log hyperparameters (if desired).
# log_hyperparameters(trainer=None, model=model, cfg=CONFIG)

# Look for an existing checkpoint.
ckpt_files = sorted(run_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
start_epoch = 1
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    print(f"Found checkpoint: {latest_ckpt}")
    start_epoch = load_checkpoint(model, optimizer, latest_ckpt) + 1

# Train the model.
num_epochs = CONFIG["train"]["num_epochs"]


model.to(device)
best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for batch in train_dataloader:
        batch = [b.to(device) for b in batch]
        loss = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # Run validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = [b.to(device) for b in batch]
            loss = model.validation_step(batch)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)
    print(
        f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
    )

    # (Optional) Save checkpoint if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        ckpt_path = run_dir / f"epoch={epoch}_val_loss={avg_val_loss:.4f}.pt"
        save_checkpoint(model, optimizer, epoch, ckpt_path)


model.eval()
test_losses = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = [b.to(device) for b in batch]
        loss = model.test_step(batch)
        test_losses.append(loss.item())
avg_test_loss = np.mean(test_losses)
print(f"Test Loss: {avg_test_loss:.4f}")
