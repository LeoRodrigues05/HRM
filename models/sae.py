"""Sparse Autoencoder (SAE) for mechanistic interpretability.

Implements SAEs for discovering sparse, overcomplete features in HRM
hidden states (z_H / z_L).

Two activation modes:
  - L1 (default): encoder → ReLU, loss = MSE + l1_coeff * mean(|h|)
  - TopK: encoder → keep only top-K activations, loss = MSE only
    (sparsity enforced structurally, no L1 penalty needed)

Architecture:
    encoder: Linear(input_dim, dict_size) + activation
    decoder: Linear(dict_size, input_dim) with unit-norm columns

Reference: Sharkey et al. (2022), Bricken et al. (2023), Gao et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SparseAutoencoder(nn.Module):
    """Single-layer SAE with L1 regularization for mechanistic interpretability.

    Architecture:
        encoder: Linear(input_dim, dict_size) + ReLU
        decoder: Linear(dict_size, input_dim)  (unit-norm columns)

    Loss: MSE_reconstruction + l1_coeff * L1(hidden_activations)

    Args:
        input_dim: Dimension of input activations (e.g. 512 for z_H).
        dict_size: Number of dictionary features (overcomplete basis).
        l1_coeff: Coefficient for L1 sparsity penalty on hidden activations.
    """

    def __init__(self, input_dim: int = 512, dict_size: int = 2048,
                 l1_coeff: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.l1_coeff = l1_coeff

        # Encoder: input → overcomplete hidden
        self.encoder = nn.Linear(input_dim, dict_size)
        # Decoder: overcomplete hidden → reconstruction
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

        # Learnable bias subtracted from input before encoding (centering)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Fixed (non-learnable) activation mean for mean-centering.
        # Defaults to zeros (no-op, backward compatible). When set via set_mean()
        # this subtracts the empirical per-dimension mean before encoding and adds
        # it back in decode, addressing dimension-level activation shift that kills
        # features at init (Simon, Adams & Zou, arXiv:2605.31518). Registered as a
        # buffer so it is saved/loaded with the model and the encode/decode
        # interface keeps operating on raw (un-centered) activations.
        self.register_buffer('act_mean', torch.zeros(input_dim))

        # Initialization
        # Kaiming (He) init for encoder — designed for ReLU activations.
        # Uses fan_in to scale weights such that the variance of activations
        # is maintained through the ReLU nonlinearity (prevents vanishing/exploding).
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        # Orthogonal init for decoder — produces a (semi-)orthogonal weight matrix.
        # This ensures decoder columns start as diverse, uncorrelated directions in
        # input space, providing a good initial dictionary for reconstruction.
        # After init, columns are normalized to unit norm.
        nn.init.orthogonal_(self.decoder.weight)

        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self._normalize_decoder()

        # --- Tracking statistics (not saved as parameters) ---
        # Track how many consecutive batches each feature has been dead
        self.register_buffer('dead_counter', torch.zeros(dict_size, dtype=torch.long))
        # Running mean activation per feature
        self.register_buffer('feature_activation_sum', torch.zeros(dict_size))
        self.register_buffer('feature_activation_count', torch.zeros(1, dtype=torch.long))
        # Number of inputs where each feature is active (for sparsity tracking)
        self.register_buffer('feature_fire_count', torch.zeros(dict_size, dtype=torch.long))

    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        # decoder.weight shape: [input_dim, dict_size]
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def set_mean(self, mean: torch.Tensor):
        """Set the fixed mean-centering vector.

        Args:
            mean: [input_dim] empirical per-dimension activation mean. Subtracted
                before encoding and added back in decode (see act_mean docstring).
        """
        with torch.no_grad():
            self.act_mean.copy_(mean.detach().to(self.act_mean.device, self.act_mean.dtype).view(-1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse hidden activations.

        Args:
            x: [*, input_dim] input activations

        Returns:
            h: [*, dict_size] sparse hidden activations (post-ReLU)
        """
        x_centered = x - self.act_mean - self.pre_bias
        h = F.relu(self.encoder(x_centered))
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden activations back to input space.

        Args:
            h: [*, dict_size] hidden activations

        Returns:
            x_hat: [*, input_dim] reconstruction
        """
        return self.decoder(h) + self.pre_bias + self.act_mean

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass: encode → decode, compute losses.

        Args:
            x: [batch_size, input_dim] input activations

        Returns:
            x_hat: [batch_size, input_dim] reconstruction
            h: [batch_size, dict_size] hidden activations
            loss_dict: dict with 'loss', 'reconstruction_loss', 'l1_loss'
        """
        # Normalize decoder columns on each forward pass
        with torch.no_grad():
            self._normalize_decoder()

        h = self.encode(x)
        x_hat = self.decode(h)

        # Losses
        reconstruction_loss = F.mse_loss(x_hat, x)
        l1_loss = h.abs().mean()
        total_loss = reconstruction_loss + self.l1_coeff * l1_loss

        loss_dict = {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'l1_loss': l1_loss,
        }

        # Update tracking statistics (no grad)
        with torch.no_grad():
            active = (h > 0)  # [batch, dict_size]
            any_active = active.any(dim=0)  # [dict_size]
            # Increment dead counter for features that didn't fire; reset for those that did
            self.dead_counter = torch.where(any_active, torch.zeros_like(self.dead_counter),
                                            self.dead_counter + 1)
            # Accumulate activation statistics
            self.feature_activation_sum += h.sum(dim=0)
            self.feature_activation_count += x.shape[0]
            self.feature_fire_count += active.long().sum(dim=0)

        return x_hat, h, loss_dict

    def get_alive_features(self, dead_threshold: int = 1000) -> torch.Tensor:
        """Return boolean mask of features that have activated recently.

        Args:
            dead_threshold: A feature is considered dead if it hasn't fired
                in this many consecutive batches.

        Returns:
            alive: [dict_size] boolean tensor
        """
        return self.dead_counter < dead_threshold

    def get_feature_stats(self, dead_threshold: int = 1000) -> Dict[str, float]:
        """Compute summary statistics about feature usage.

        Returns:
            dict with: alive_count, alive_frac, dead_count, mean_sparsity,
                       mean_activation, L0 (mean active features per input)
        """
        alive = self.get_alive_features(dead_threshold)
        total_inputs = max(self.feature_activation_count.item(), 1)

        # Mean sparsity: fraction of inputs where each alive feature fires
        sparsity_per_feature = self.feature_fire_count.float() / total_inputs
        mean_sparsity = sparsity_per_feature[alive].mean().item() if alive.any() else 0.0

        # L0: mean number of active features per input
        l0 = self.feature_fire_count.float().sum().item() / total_inputs

        # Mean activation (across all features and inputs)
        mean_act = self.feature_activation_sum.sum().item() / (total_inputs * self.dict_size)

        return {
            'alive_count': int(alive.sum().item()),
            'alive_frac': alive.float().mean().item(),
            'dead_count': int((~alive).sum().item()),
            'mean_sparsity': mean_sparsity,
            'mean_activation': mean_act,
            'L0': l0,
        }

    def reset_stats(self):
        """Reset all tracking statistics."""
        self.dead_counter.zero_()
        self.feature_activation_sum.zero_()
        self.feature_activation_count.zero_()
        self.feature_fire_count.zero_()

    def reinitialize_dead_features(self, data: torch.Tensor, dead_threshold: int = 1000):
        """Re-initialize dead features near the highest-error inputs.

        For each dead feature, set its encoder weights to point toward the
        input sample with the highest reconstruction error, plus small noise.
        This gives dead features a chance to learn useful directions.

        Args:
            data: [N, input_dim] sample of training data for computing errors.
            dead_threshold: Features dead for this many batches get reinitialized.
        """
        dead_mask = ~self.get_alive_features(dead_threshold)
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        with torch.no_grad():
            # Find inputs with highest reconstruction error
            x_hat, _, _ = self.forward(data)
            errors = (data - x_hat).pow(2).sum(dim=-1)  # [N]

            # Use top-n_dead error inputs (with replacement if needed)
            n_samples = min(n_dead, data.shape[0])
            _, top_indices = errors.topk(n_samples)

            # Cycle through top-error inputs for each dead feature
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            for i, feat_idx in enumerate(dead_indices):
                sample_idx = top_indices[i % n_samples]
                sample = data[sample_idx]

                # Set encoder weight to point toward this sample (centered)
                direction = sample - self.act_mean - self.pre_bias
                direction = direction / direction.norm().clamp(min=1e-8)
                self.encoder.weight[feat_idx] = direction * 0.1  # small scale
                self.encoder.bias[feat_idx] = 0.0

                # Set decoder column to same direction
                self.decoder.weight[:, feat_idx] = direction

            # Reset dead counters for reinitialized features
            self.dead_counter[dead_mask] = 0

            # Re-normalize decoder
            self._normalize_decoder()

        return n_dead


class TopKSparseAutoencoder(SparseAutoencoder):
    """SAE with TopK activation function instead of L1 penalty.

    Instead of ReLU + L1 penalty, only the top-K activations are kept
    nonzero per input. This directly enforces a fixed sparsity level
    (L0 = K) without needing to tune an L1 coefficient.

    Reference: Gao et al. (2024) "Scaling and evaluating sparse autoencoders"

    Args:
        input_dim: Dimension of input activations.
        dict_size: Number of dictionary features.
        k: Number of top activations to keep per input.
    """

    def __init__(self, input_dim: int = 512, dict_size: int = 2048,
                 k: int = 64):
        # l1_coeff=0 since TopK doesn't use L1 penalty
        super().__init__(input_dim=input_dim, dict_size=dict_size, l1_coeff=0.0)
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input, keeping only the top-K activations.

        Args:
            x: [*, input_dim] input activations

        Returns:
            h: [*, dict_size] sparse hidden activations (only K nonzero per row)
        """
        x_centered = x - self.act_mean - self.pre_bias
        pre_act = self.encoder(x_centered)  # [*, dict_size]

        # Keep only top-K values, zero the rest
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)  # [*, K]
        topk_vals = F.relu(topk_vals)  # ensure non-negative

        h = torch.zeros_like(pre_act)
        h.scatter_(-1, topk_idx, topk_vals)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass with TopK activation.

        Loss is MSE only — sparsity is enforced by the TopK structure.
        The l1_loss key is still returned (as 0) for interface compatibility.
        """
        with torch.no_grad():
            self._normalize_decoder()

        h = self.encode(x)
        x_hat = self.decode(h)

        reconstruction_loss = F.mse_loss(x_hat, x)
        l1_loss = torch.tensor(0.0, device=x.device)  # not used, for compat
        total_loss = reconstruction_loss

        loss_dict = {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'l1_loss': l1_loss,
        }

        with torch.no_grad():
            active = (h > 0)
            any_active = active.any(dim=0)
            self.dead_counter = torch.where(any_active, torch.zeros_like(self.dead_counter),
                                            self.dead_counter + 1)
            self.feature_activation_sum += h.sum(dim=0)
            self.feature_activation_count += x.shape[0]
            self.feature_fire_count += active.long().sum(dim=0)

        return x_hat, h, loss_dict
