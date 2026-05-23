"""MCTS-based AI Agent."""

import torch

from game.board import Move, QuoridorState
from mcts.search import run_mcts
from model.network import QuoridorNet
from utils.config import MCTSConfig


class MCTSAgent:
    """Agent that selects moves using MCTS and a neural network."""

    def __init__(self, network: QuoridorNet, config: MCTSConfig, device: torch.device):
        self.network = network
        self.config = config
        self.device = device
        self.network.eval()

    def select_move(self, state: QuoridorState, training: bool = False) -> Move:
        probs = run_mcts(
            initial_state=state,
            network=self.network,
            config=self.config,
            device=self.device,
            training=training,
        )

        if training:
            # Sample from probability distribution
            moves = list(probs.keys())
            probabilities = list(probs.values())

            # Simple python standard lib choice with weights
            import random
            return random.choices(moves, weights=probabilities, k=1)[0]
        else:
            # Deterministic selection (run_mcts handles this if temp=0, but just in case)
            best_move = max(probs.items(), key=lambda item: item[1])[0]
            return best_move
