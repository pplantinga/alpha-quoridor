"""Experience replay buffer for self-play generated data."""

import collections
import random

import torch

Experience = tuple[torch.Tensor, torch.Tensor, float]


class ExperienceBuffer:
    """A circular buffer for storing experience from self-play."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: collections.deque[Experience] = collections.deque(maxlen=max_size)

    def add(self, state_tensor: torch.Tensor, policy_target: torch.Tensor, value_target: float) -> None:
        """Add a single experience tuple to the buffer."""
        self.buffer.append((state_tensor, policy_target, value_target))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences.

        Returns:
            states: Tensor of shape (B, 7, N, N)
            policies: Tensor of shape (B, 3*N*N)
            values: Tensor of shape (B, 1)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Have {len(self.buffer)}, need {batch_size}")

        batch = random.sample(self.buffer, batch_size)

        states, policies, values = zip(*batch)

        state_batch = torch.stack(states)
        policy_batch = torch.stack(policies)
        value_batch = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        return state_batch, policy_batch, value_batch

    def __len__(self) -> int:
        return len(self.buffer)
