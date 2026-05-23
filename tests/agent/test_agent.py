"""Tests for agent implementations."""

import torch

from agent.mcts_agent import MCTSAgent
from agent.random_agent import RandomAgent
from game.board import initial_state
from game.rules import legal_moves
from model.network import QuoridorNet
from utils.config import MCTSConfig, ModelConfig


def test_random_agent() -> None:
    state = initial_state(5, 3)
    agent = RandomAgent()
    move = agent.select_move(state)
    assert move in legal_moves(state)


def test_mcts_agent() -> None:
    board_size = 3
    state = initial_state(board_size, 1)

    # Tiny network
    model_config = ModelConfig(hidden_dim=2, num_blocks=1)
    network = QuoridorNet(board_size, model_config)

    mcts_config = MCTSConfig(num_simulations=5)
    agent = MCTSAgent(network, mcts_config, torch.device("cpu"))

    move = agent.select_move(state, training=True)
    assert move in legal_moves(state)

    move_eval = agent.select_move(state, training=False)
    assert move_eval in legal_moves(state)
