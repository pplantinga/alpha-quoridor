"""Checkpoint loading utilities."""

import torch.nn as nn


def load_model_weights(model: nn.Module, state_dict: dict) -> None:
    """Load state_dict into model, handling _orig_mod. prefix from torch.compile."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
