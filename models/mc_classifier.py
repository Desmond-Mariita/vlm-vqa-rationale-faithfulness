"""Multiple-choice classification head for VQA answer prediction."""

from __future__ import annotations
import torch
import torch.nn as nn


class MCClassifier(nn.Module):
    """Simple 4-way classification head for multiple-choice VQA.

    Architecture: Linear -> GELU -> Dropout -> Linear, mapping pooled
    hidden states of shape ``(B, H)`` to logits of shape ``(B, num_classes)``.
    """

    def __init__(self, input_dim: int, num_classes: int = 4, p_drop: float = 0.1):
        """Initialise the classifier head.

        Args:
            input_dim: Dimensionality of the pooled hidden-state input.
            num_classes: Number of output classes (answer choices).
            p_drop: Dropout probability applied after the GELU activation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits from pooled hidden states.

        Args:
            x: Tensor of shape ``(B, input_dim)`` containing pooled
                hidden-state representations.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.net(x)
