#!/usr/bin/env python3
"""
This file defines two models (CrystGNN_Supervise and CDVAE) without using
PyTorch Lightning or Hydra. All hyperparameter configurations are defined
at the top of the file.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

# (Assuming these modules are available from your cdvae package)
from cdvae.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)
from cdvae.common.utils import PROJECT_ROOT
from cdvae.models.embeddings import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM

###############################################################################
# Configuration and Helper Classes/Functions
###############################################################################


class DotDict(dict):
    """A simple dict subclass supporting attribute access."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def instantiate_from_config(cfg: Dict, **extra_kwargs):
    """
    Mimics hydra.utils.instantiate: expects a dict with key "class" (the class to instantiate)
    and an optional key "params" (a dict of keyword arguments).
    Additional keyword arguments (extra_kwargs) are passed to the class constructor.
    """
    cls = cfg["class"]
    params = cfg.get("params", {}).copy()
    params.update(extra_kwargs)
    return cls(**params)


def instantiate_optimizer(optim_cfg: Dict, parameters):
    """
    Instantiate an optimizer from the configuration.
    The config should contain:
      - "optimizer": the optimizer class (e.g. torch.optim.Adam)
      - "optimizer_params": a dict of parameters for the optimizer.
      - "use_lr_scheduler": bool, whether to also create a learning rate scheduler.
      - Optionally, "lr_scheduler": the LR scheduler class and "lr_scheduler_params".
    """
    opt_class = optim_cfg["optimizer"]
    opt_params = optim_cfg.get("optimizer_params", {})
    optimizer = opt_class(parameters, **opt_params)
    if not optim_cfg.get("use_lr_scheduler", False):
        return optimizer
    else:
        scheduler_class = optim_cfg["lr_scheduler"]
        scheduler_params = optim_cfg.get("lr_scheduler_params", {})
        scheduler = scheduler_class(optimizer, **scheduler_params)
        return optimizer, scheduler


###############################################################################
# Hyperparameter Configuration
###############################################################################

CONFIG = DotDict(
    {
        "optim": {
            "optimizer": torch.optim.Adam,
            "optimizer_params": {"lr": 1e-3},
            "use_lr_scheduler": False,
            # If you wish to use an LR scheduler, e.g.:
            # "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            # "lr_scheduler_params": {"step_size": 10, "gamma": 0.1},
        },
        "data": {
            "prop": "scaled_lattice",
            "lattice_scale_method": "scale_length",
        },
        "logging": {},
        # In a real-case scenario, replace DummyEncoder/Decoder with your implementations.
        "encoder": {
            "class": None,  # <-- set below after DummyEncoder is defined.
            "params": {},  # any additional parameters for the encoder
        },
        "decoder": {
            "class": None,  # <-- set below after DummyDecoder is defined.
            "params": {},  # any additional parameters for the decoder
        },
        "latent_dim": 128,
        "hidden_dim": 256,
        "fc_num_layers": 3,
        "max_atoms": 100,
        "predict_property": True,
        "teacher_forcing_lattice": False,
        "teacher_forcing_max_epoch": 10,
        "sigma_begin": 0.1,
        "sigma_end": 1.0,
        "num_noise_level": 10,
        "type_sigma_begin": 0.1,
        "type_sigma_end": 1.0,
        "cost_natom": 1.0,
        "cost_lattice": 1.0,
        "cost_coord": 1.0,
        "cost_type": 1.0,
        "beta": 1.0,
        "cost_composition": 1.0,
        "cost_property": 1.0,
    }
)

###############################################################################
# Dummy Encoder/Decoder Implementations (Replace with your own!)
###############################################################################


class DummyEncoder(nn.Module):
    def __init__(self, num_targets=None):
        super().__init__()
        # A dummy linear layer – in practice, your encoder will be more complex.
        self.linear = nn.Linear(10, 10)

    def forward(self, batch):
        # Assume batch has an attribute "x" (or replace with your own logic)
        x = batch.x if hasattr(batch, "x") else torch.rand(1, 10)
        return self.linear(x)


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # A dummy linear layer – in practice, your decoder will be more complex.
        self.linear = nn.Linear(10, 10)

    def forward(self, z, frac_coords, atom_types, num_atoms, lengths, angles):
        # Produce dummy outputs with the expected shapes.
        # For example, pred_cart_coord_diff should be of shape (N, 3)
        # and pred_atom_types of shape (N, MAX_ATOMIC_NUM)
        N = frac_coords.shape[0]
        pred_cart_coord_diff = torch.zeros((N, 3), device=z.device)
        pred_atom_types = torch.zeros((N, MAX_ATOMIC_NUM), device=z.device)
        return pred_cart_coord_diff, pred_atom_types


# Update the CONFIG to use the dummy implementations.
CONFIG.encoder["class"] = DummyEncoder
CONFIG.decoder["class"] = DummyDecoder

###############################################################################
# Utility Function: Build an MLP
###############################################################################


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


###############################################################################
# Base Module (No Lightning)
###############################################################################


class BaseModule(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        # Save configuration as an attribute.
        self.hparams = DotDict(config)
        # For tracking epoch (used in teacher forcing logic)
        self.current_epoch = 0
        # Determine device (can be overridden later)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure_optimizers(self):
        """Manually instantiate optimizer (and scheduler, if desired)."""
        return instantiate_optimizer(self.hparams.optim, self.parameters())


###############################################################################
# Model: CrystGNN_Supervise
###############################################################################


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, **config) -> None:
        super().__init__(**config)
        # Instantiate encoder from configuration.
        self.encoder = instantiate_from_config(self.hparams.encoder)

    def forward(self, batch) -> torch.Tensor:
        preds = self.encoder(batch)  # shape (N, 1) expected
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y)
        # Instead of self.log_dict (Lightning logging), we simply print.
        print({"train_loss": loss.item()})
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y)
        # Compute additional stats if needed…
        print({"val_loss": loss.item()})
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        loss = F.mse_loss(preds, batch.y)
        print({"test_loss": loss.item()})
        return loss


###############################################################################
# Model: CDVAE
###############################################################################


class CDVAE(BaseModule):
    def __init__(self, **config) -> None:
        super().__init__(**config)
        # Instantiate encoder and decoder using the configuration.
        self.encoder = instantiate_from_config(
            self.hparams.encoder, num_targets=self.hparams.latent_dim
        )
        self.decoder = instantiate_from_config(self.hparams.decoder)
        # Define additional layers.
        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_atoms + 1,
        )
        self.fc_lattice = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            6,
        )
        self.fc_composition = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            MAX_ATOMIC_NUM,
        )
        if self.hparams.predict_property:
            self.fc_property = build_mlp(
                self.hparams.latent_dim,
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers,
                1,
            )

        # Create noise level parameters.
        sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.sigma_begin),
                    np.log(self.hparams.sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
            dtype=torch.float32,
        )
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.type_sigma_begin),
                    np.log(self.hparams.type_sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
            dtype=torch.float32,
        )
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        # Create an embedding from KHOT_EMBEDDINGS.
        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # These scalers would normally be set externally (e.g., via a data module).
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(
        self,
        z,
        gt_num_atoms=None,
        gt_lengths=None,
        gt_angles=None,
        teacher_forcing=False,
    ):
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = self.predict_lattice(z, num_atoms)
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms
        )
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # Obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # Initialize fractional coordinates.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # Annealed Langevin dynamics.
        for sigma in tqdm(
            self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar
        ):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(
                    step_size * 2
                )
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles
                )
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms
                )
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = (
                    cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                )
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms
                )

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {
            "num_atoms": num_atoms,
            "lengths": lengths,
            "angles": angles,
            "frac_coords": cur_frac_coords,
            "atom_types": cur_atom_types,
            "is_traj": False,
        }

        if ld_kwargs.save_traj:
            output_dict.update(
                dict(
                    all_frac_coords=torch.stack(all_frac_coords, dim=0),
                    all_atom_types=torch.stack(all_atom_types, dim=0),
                    all_pred_cart_coord_diff=torch.stack(
                        all_pred_cart_coord_diff, dim=0
                    ),
                    all_noise_cart=torch.stack(all_noise_cart, dim=0),
                    is_traj=True,
                )
            )
        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim, device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing, training):
        mu, log_var, z = self.encode(batch)
        (
            pred_num_atoms,
            pred_lengths_and_angles,
            pred_lengths,
            pred_angles,
            pred_composition_per_atom,
        ) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing
        )

        # Sample noise levels.
        noise_level = torch.randint(
            0, self.sigmas.size(0), (batch.num_atoms.size(0),), device=self.device
        )
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0
        )

        type_noise_level = torch.randint(
            0, self.type_sigmas.size(0), (batch.num_atoms.size(0),), device=self.device
        )
        used_type_sigmas_per_atom = self.type_sigmas[
            type_noise_level
        ].repeat_interleave(batch.num_atoms, dim=0)

        # Add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM)
            + pred_composition_probs * used_type_sigmas_per_atom[:, None]
        )
        rand_atom_types = (
            torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 1
        )

        # Add noise to the Cartesian coordinates.
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) * used_sigmas_per_atom[:, None]
        )
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms
        )
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms
        )

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z,
            noisy_frac_coords,
            rand_atom_types,
            batch.num_atoms,
            pred_lengths,
            pred_angles,
        )

        # Compute individual loss components.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch
        )
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch
        )
        type_loss = self.type_loss(
            pred_atom_types, batch.atom_types, used_type_sigmas_per_atom, batch
        )
        kld_loss = self.kld_loss(mu, log_var)
        property_loss = (
            self.property_loss(z, batch) if self.hparams.predict_property else 0.0
        )

        return {
            "num_atom_loss": num_atom_loss,
            "lattice_loss": lattice_loss,
            "composition_loss": composition_loss,
            "coord_loss": coord_loss,
            "type_loss": type_loss,
            "kld_loss": kld_loss,
            "property_loss": property_loss,
            "pred_num_atoms": pred_num_atoms,
            "pred_lengths_and_angles": pred_lengths_and_angles,
            "pred_lengths": pred_lengths,
            "pred_angles": pred_angles,
            "pred_cart_coord_diff": pred_cart_coord_diff,
            "pred_atom_types": pred_atom_types,
            "pred_composition_per_atom": pred_composition_per_atom,
            "target_frac_coords": batch.frac_coords,
            "target_atom_types": batch.atom_types,
            "rand_frac_coords": noisy_frac_coords,
            "rand_atom_types": rand_atom_types,
            "z": z,
        }

    def generate_rand_init(
        self, pred_composition_per_atom, pred_lengths, pred_angles, num_atoms, batch
    ):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3, device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom, dim=-1)
        rand_atom_types = self.sample_composition(pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        batch_idx = torch.arange(
            len(num_atoms), device=num_atoms.device
        ).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch_idx.size(0)
        composition_prob = scatter(
            composition_prob, index=batch_idx, dim=0, reduce="mean"
        )
        all_sampled_comp = []
        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()
            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)
                left_comp_prob = comp_prob - comp_num.float() / num_atom
                left_comp_prob[left_comp_prob < 0.0] = 0.0
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True
                )
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)
            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)
        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        if self.scaler is not None:
            self.scaler.match_device(z)
            return self.scaler.inverse_transform(self.fc_property(z))
        else:
            return self.fc_property(z)

    def predict_lattice(self, z, num_atoms):
        if self.lattice_scaler is not None:
            self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)
        if self.lattice_scaler is not None:
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles
            )
        else:
            scaled_preds = pred_lengths_and_angles
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == "scale_length":
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        if self.lattice_scaler is not None:
            self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == "scale_length":
            target_lengths = batch.lengths / batch.num_atoms.view(-1, 1).float() ** (
                1 / 3
            )
        else:
            target_lengths = batch.lengths
        target_lengths_and_angles = torch.cat([target_lengths, batch.angles], dim=-1)
        if self.lattice_scaler is not None:
            target_lengths_and_angles = self.lattice_scaler.transform(
                target_lengths_and_angles
            )
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_composition_per_atom, target_atom_types, reduction="none"
        )
        return scatter(loss, batch.batch, reduce="mean").mean()

    def coord_loss(
        self, pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch
    ):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords,
            noisy_cart_coords,
            batch.lengths,
            batch.angles,
            batch.num_atoms,
            self.device,
            return_vector=True,
        )
        target_cart_coord_diff = (
            target_cart_coord_diff / used_sigmas_per_atom[:, None] ** 2
        )
        pred_cart_coord_diff = pred_cart_coord_diff / used_sigmas_per_atom[:, None]
        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1
        )
        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce="mean").mean()

    def type_loss(
        self, pred_atom_types, target_atom_types, used_type_sigmas_per_atom, batch
    ):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_atom_types, target_atom_types, reduction="none")
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce="mean").mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = self.current_epoch <= self.hparams.teacher_forcing_max_epoch
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="train")
        print(log_dict)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="val")
        print(log_dict)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="test")
        print(log_dict)
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs["num_atom_loss"]
        lattice_loss = outputs["lattice_loss"]
        coord_loss = outputs["coord_loss"]
        type_loss = outputs["type_loss"]
        kld_loss = outputs["kld_loss"]
        composition_loss = outputs["composition_loss"]
        property_loss = outputs["property_loss"]

        total_loss = (
            self.hparams.cost_natom * num_atom_loss
            + self.hparams.cost_lattice * lattice_loss
            + self.hparams.cost_coord * coord_loss
            + self.hparams.cost_type * type_loss
            + self.hparams.beta * kld_loss
            + self.hparams.cost_composition * composition_loss
            + self.hparams.cost_property * property_loss
        )

        log_dict = {
            f"{prefix}_loss": total_loss.item(),
            f"{prefix}_natom_loss": num_atom_loss.item(),
            f"{prefix}_lattice_loss": lattice_loss.item(),
            f"{prefix}_coord_loss": coord_loss.item(),
            f"{prefix}_type_loss": type_loss.item(),
            f"{prefix}_kld_loss": kld_loss.item(),
            f"{prefix}_composition_loss": composition_loss.item(),
        }

        if prefix != "train":
            total_loss = (
                self.hparams.cost_coord * coord_loss
                + self.hparams.cost_type * type_loss
            )
            pred_num_atoms = outputs["pred_num_atoms"].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms
            ).sum() / batch.num_graphs
            pred_lengths_and_angles = outputs["pred_lengths_and_angles"]
            if self.lattice_scaler is not None:
                scaled_preds = self.lattice_scaler.inverse_transform(
                    pred_lengths_and_angles
                )
            else:
                scaled_preds = pred_lengths_and_angles
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.hparams.data.lattice_scale_method == "scale_length":
                pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float() ** (
                    1 / 3
                )
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)
            pred_atom_types = outputs["pred_atom_types"]
            target_atom_types = outputs["target_atom_types"]
            type_accuracy = pred_atom_types.argmax(dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(
                type_accuracy.float(), batch.batch, dim=0, reduce="mean"
            ).mean()
            log_dict.update(
                {
                    f"{prefix}_loss": total_loss.item(),
                    f"{prefix}_property_loss": (
                        property_loss.item()
                        if isinstance(property_loss, torch.Tensor)
                        else property_loss
                    ),
                    f"{prefix}_natom_accuracy": num_atom_accuracy.item(),
                    f"{prefix}_lengths_mard": lengths_mard,
                    f"{prefix}_angles_mae": angles_mae.item(),
                    f"{prefix}_volumes_mard": volumes_mard,
                    f"{prefix}_type_accuracy": type_accuracy.item(),
                }
            )

        return log_dict, total_loss


###############################################################################
# Main Function (Entry Point)
###############################################################################


def main():
    # For demonstration, we instantiate the CDVAE model using our CONFIG.
    # (You could also instantiate CrystGNN_Supervise by calling its constructor.)
    model = CDVAE(**CONFIG)
    model.to(model.device)
    print("Model instantiated:")
    print(model)
    # Here you would add your training loop that calls model.training_step(),
    # model.validation_step(), etc.
    # For example:
    # for epoch in range(num_epochs):
    #     model.current_epoch = epoch
    #     for batch_idx, batch in enumerate(train_loader):
    #         loss = model.training_step(batch, batch_idx)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     # Run validation, etc.
    #
    # In this refactored code, we simply print the model summary.


if __name__ == "__main__":
    main()
