"""Tests for training buffer."""

import pytest
import torch

from training.buffer import ExperienceBuffer


def test_buffer_add_and_sample() -> None:
    buffer = ExperienceBuffer(max_size=10)

    for i in range(5):
        s = torch.zeros((7, 5, 5))
        p = torch.zeros(75)
        v = float(i)
        buffer.add(s, p, v)

    assert len(buffer) == 5

    # Sample 3 items
    s_batch, p_batch, v_batch = buffer.sample(3)
    assert s_batch.shape == (3, 7, 5, 5)
    assert p_batch.shape == (3, 75)
    assert v_batch.shape == (3, 1)


def test_buffer_eviction() -> None:
    buffer = ExperienceBuffer(max_size=3)

    for i in range(5):
        s = torch.zeros((7, 5, 5))
        p = torch.zeros(75)
        buffer.add(s, p, float(i))

    assert len(buffer) == 3

    # we added 0, 1, 2, 3, 4
    # Max size = 3, so buffer has 2, 3, 4
    _s_batch, _p_batch, v_batch = buffer.sample(3)
    values = set(v_batch.squeeze().tolist())
    assert values == {2.0, 3.0, 4.0}


def test_buffer_sample_too_many() -> None:
    buffer = ExperienceBuffer(max_size=10)
    buffer.add(torch.zeros((7, 5, 5)), torch.zeros(75), 1.0)

    with pytest.raises(ValueError):
        buffer.sample(5)
