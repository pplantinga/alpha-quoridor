"""Verification script for Minimax Agent."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from agent.minimax_agent import MinimaxAgent
from agent.random_agent import RandomAgent
from game.board import initial_state
from game.rules import apply_move


def play_game(agent0, agent1, board_size=9, walls=10):
    state = initial_state(board_size, walls)
    agents = [agent0, agent1]

    move_count = 0
    # Add a move limit to prevent infinite loops in edge cases
    while not state.is_terminal and move_count < 300:
        agent = agents[state.current_player]
        move = agent.select_move(state)
        state = apply_move(state, move)
        move_count += 1

    return state.winner, move_count

def main():
    num_games = 10
    minimax_depth = 2

    minimax_wins = 0
    random_wins = 0
    draws = 0

    print(f"--- Verifying MinimaxAgent (depth={minimax_depth}) vs RandomAgent ---")

    # Half games as P0
    for i in range(num_games // 2):
        print(f"Game {i+1}: Minimax(P0) vs Random(P1)... ", end="", flush=True)
        winner, moves = play_game(MinimaxAgent(depth=minimax_depth), RandomAgent())
        if winner == 0:
            minimax_wins += 1
            print(f"Minimax wins in {moves} moves")
        elif winner == 1:
            random_wins += 1
            print(f"Random wins in {moves} moves")
        else:
            draws += 1
            print("Draw")

    # Half games as P1
    for i in range(num_games // 2):
        print(f"Game {i + num_games//2 + 1}: Random(P0) vs Minimax(P1)... ", end="", flush=True)
        winner, moves = play_game(RandomAgent(), MinimaxAgent(depth=minimax_depth))
        if winner == 1:
            minimax_wins += 1
            print(f"Minimax wins in {moves} moves")
        elif winner == 0:
            random_wins += 1
            print(f"Random wins in {moves} moves")
        else:
            draws += 1
            print("Draw")

    print("\n--- Final Results ---")
    print(f"Minimax Wins: {minimax_wins}")
    print(f"Random Wins:  {random_wins}")
    print(f"Draws:        {draws}")

    if minimax_wins > random_wins:
        print("\nSUCCESS: MinimaxAgent outperformed RandomAgent.")
    else:
        print("\nFAILURE: MinimaxAgent did not outperform RandomAgent.")

if __name__ == "__main__":
    main()
