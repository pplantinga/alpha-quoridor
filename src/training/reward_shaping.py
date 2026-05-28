"""Shared reward-shaping helpers for Alpha Quoridor training.

Used by both self_play.py (MCTS self-play) and bootstrap_utils.py (minimax priming)
so the shaped value targets are computed identically in both data sources.
"""

import torch


def heuristic_value(s_enc: torch.Tensor) -> float:
    """Minimax-style board evaluation from a canonical 9-channel encoded state.

    Mirrors ``MinimaxAgent.evaluate()``:
      * Positive → current player is winning (shorter path, more walls).
      * Negative → current player is losing.

    All inputs come from the *canonical* encoding where channel 0 = current
    player, channel 1 = opponent, channels 7/8 = distance maps (normalized
    by board-size ``n``), channels 5/6 = walls remaining (normalized by
    ``walls_per_player``).

    Returns a float in roughly [-1.1, 1.1] (clamped before storing to buffer).
    """
    my_pos_flat  = int(torch.argmax(s_enc[0].view(-1)).item())
    opp_pos_flat = int(torch.argmax(s_enc[1].view(-1)).item())

    # Distance to own goal (ch7 = mine, ch8 = opponent's)
    my_dist  = float(s_enc[7].view(-1)[my_pos_flat].item())
    opp_dist = float(s_enc[8].view(-1)[opp_pos_flat].item())

    # Walls remaining (already normalized to [0, 1])
    my_walls  = float(s_enc[5, 0, 0].item())
    opp_walls = float(s_enc[6, 0, 0].item())

    # Distance advantage: positive means I am closer to my goal than opponent
    dist_advantage = opp_dist - my_dist

    # Small wall bonus matching minimax (0.5 wall-counts map to 0.1 here because
    # walls are already normalized)
    wall_advantage = 0.1 * (my_walls - opp_walls)

    return dist_advantage + wall_advantage


def step_shaping(
    s_enc: torch.Tensor,
    next_enc: torch.Tensor,
    progress_weight: float,
    block_weight: float,
) -> float:
    """Shaped reward for a single transition s_enc → next_enc.

    Captures two independent signals:
    1. **My progress** -- how much my path to goal shortened.
    2. **Opponent blocked** -- how much the opponent's path to goal lengthened
       (non-zero even for pure wall placements that don't move our pawn).

    The canonical encoding perspective flips on each turn:
    * In ``s_enc``   (my turn):   ch0 = me, ch7 = my dist,  ch8 = opp dist
    * In ``next_enc`` (opp turn): ch0 = opp, ch7 = opp dist, ch8 = my dist

    Args:
        s_enc:          Encoded state *before* my move (shape 9, N, N).
        next_enc:       Encoded state *after* my move (shape 9, N, N).
        progress_weight: Weight for own path shortening.  Suggested: 0.3.
        block_weight:   Weight for opponent path lengthening.  Suggested: 0.2.

    Returns:
        A float shaping bonus, positive when the move was strategically good.
    """
    # My position in s_enc (channel 0 = current player)
    my_pos_flat   = int(torch.argmax(s_enc[0].view(-1)).item())
    # Opponent position in s_enc (channel 1 = opponent)
    opp_pos_flat  = int(torch.argmax(s_enc[1].view(-1)).item())

    # My distance before the move (ch7 = current player's dist map)
    my_dist_before  = float(s_enc[7].view(-1)[my_pos_flat].item())
    # Opponent distance before the move (ch8 = opponent's dist map)
    opp_dist_before = float(s_enc[8].view(-1)[opp_pos_flat].item())

    # After the move the roles flip: I am now the "opponent" in next_enc.
    # My new position is in channel 1 of next_enc.
    my_next_pos_flat  = int(torch.argmax(next_enc[1].view(-1)).item())
    # Opponent's new position is in channel 0 of next_enc.
    opp_next_pos_flat = int(torch.argmax(next_enc[0].view(-1)).item())

    # My distance after (ch8 = "opponent" dist in next_enc = me)
    my_dist_after   = float(next_enc[8].view(-1)[my_next_pos_flat].item())
    # Opponent's distance after (ch7 = "current player" dist in next_enc = opp)
    opp_dist_after  = float(next_enc[7].view(-1)[opp_next_pos_flat].item())

    # Positive → I moved closer to my goal
    my_progress = my_dist_before - my_dist_after
    # Positive → opponent's path got longer (good for me)
    opp_blocked = opp_dist_after - opp_dist_before

    return progress_weight * my_progress + block_weight * opp_blocked
