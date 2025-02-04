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

# (Assuming these exist in your codebase)
from cdvae.common.utils import PROJECT_ROOT, log_hyperparameters
from cdvae.models.cdvae_model import CDVAE, hparams

###############################################################################
# CONFIGURATION
###############################################################################
# All hyperparameters and configuration options are stored in the CONFIG dict.
CONFIG = OmegaConf.create(
    {
        "train": {
            "deterministic": True,
            "random_seed": 42,
            "fast_dev_run": False,  # Debug mode (if True, use CPU and no multiprocessing)
            "num_epochs": 100,
            "monitor_metric": "val_loss",
            "monitor_metric_mode": "min",
            "early_stopping": {
                "patience": 5,
                "verbose": True,
            },
            "model_checkpoints": {
                "save_top_k": 1,
                "verbose": True,
            },
            "batch_size": 32,
            "learning_rate": 1e-3,
        },
        "logging": {
            "val_check_interval": 1,
            "progress_bar_refresh_rate": 20,
            # (WandB settings are not used in this plain script version.)
            "wandb": {
                "mode": "online",
                "project": "my_project",
            },
            "wandb_watch": {
                "log": "all",
                "log_freq": 100,
            },
        },
        "data": {
            "datamodule": {
                # Replace with your actual data module class name or import path.
                "_target_": "MyDataModule",
                "num_workers": {"train": 4, "val": 4, "test": 4},
                "batch_size": 32,
            },
        },
        "model": {
            # Replace with your actual model class name or import path.
            "_target_": "MyModel",
            # Add any model-specific parameters here.
        },
        "optim": {
            "optimizer": "Adam",  # interpreted as torch.optim.Adam
            "optimizer_params": {"lr": 1e-3},
        },
        "core": {
            "tags": ["experiment", "test"],
        },
    }
)


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
# DUMMY DATA MODULE & MODEL (Replace these with your actual implementations)
###############################################################################


###############################################################################
# TRAINING & TESTING LOOPS
###############################################################################
def train_loop(
    model, optimizer, datamodule, num_epochs: int, device, checkpoint_dir: Path
):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in datamodule.train_dataloader():
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
            for batch in datamodule.val_dataloader():
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
            ckpt_path = checkpoint_dir / f"epoch={epoch}_val_loss={avg_val_loss:.4f}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)


def test_loop(model, datamodule, device):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            batch = [b.to(device) for b in batch]
            loss = model.test_step(batch)
            test_losses.append(loss.item())
    avg_test_loss = np.mean(test_losses)
    print(f"Test Loss: {avg_test_loss:.4f}")


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
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
    dm_cfg = CONFIG["data"]["datamodule"]
    datamodule = MyDataModule(
        batch_size=dm_cfg["batch_size"],
        num_workers=dm_cfg["num_workers"],
    )

    # Instantiate model.
    model = CDVAE(hparams=hparams)

    # (Pass scalers from datamodule to model, if applicable.)
    # Here we assume both are dicts; in your code these could be objects with a copy() method.
    model.lattice_scaler = (
        datamodule.lattice_scaler.copy()
        if isinstance(datamodule.lattice_scaler, dict)
        else datamodule.lattice_scaler
    )
    model.scaler = (
        datamodule.scaler.copy()
        if isinstance(datamodule.scaler, dict)
        else datamodule.scaler
    )

    # Save scalers.
    torch.save(datamodule.lattice_scaler, run_dir / "lattice_scaler.pt")
    torch.save(datamodule.scaler, run_dir / "prop_scaler.pt")

    # Instantiate optimizer.
    optim_cfg = CONFIG["optim"]
    optimizer = torch.optim.Adam(model.parameters(), **optim_cfg["optimizer_params"])

    # Log hyperparameters (if desired).
    log_hyperparameters(trainer=None, model=model, cfg=CONFIG)

    # Look for an existing checkpoint.
    ckpt_files = sorted(run_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
    start_epoch = 1
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        print(f"Found checkpoint: {latest_ckpt}")
        start_epoch = load_checkpoint(model, optimizer, latest_ckpt) + 1

    # Train the model.
    num_epochs = CONFIG["train"]["num_epochs"]
    train_loop(model, optimizer, datamodule, num_epochs, device, run_dir)

    # Test the model.
    test_loop(model, datamodule, device)


if __name__ == "__main__":
    main()
