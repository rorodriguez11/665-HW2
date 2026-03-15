# Authors: S. El Alaoui and ChatGPT 5.2
# benchmark.py
#
# Usage (from the same folder as coinline.py):
#   python benchmark.py
#   python benchmark.py --trials 30 --ncoins 15 --seed 0
#
# This runs MCTS (as 'player') vs minimax (as 'ai') on many random coin lines.
# For each coin line it plays TWO games:
#   Game A: MCTS starts (state.turn='player')
#   Game B: Minimax starts (state.turn='ai')
#
# It reports, for each MCTS budget:
#   - win rate of MCTS
#   - average score difference (MCTS - minimax)
#   - average MCTS move time (ms/move)
# (Optionally also minimax ms/move)

from __future__ import annotations

import argparse
import random
import time
import statistics as stats
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import coinline as cl


# ----------------------------
# Config
# ----------------------------

DEFAULT_BUDGETS = [20, 200, 2000, 20000]
DEFAULT_REWARD_MODES = ["winloss"]
DEFAULT_C = 2 ** 0.5  # sqrt(2)
DEFAULT_NCOINS = 15
DEFAULT_TRIALS = 30


# ----------------------------
# Helpers / Metrics
# ----------------------------

@dataclass
class GameResult:
    mcts_wins: int
    ties: int
    minimax_wins: int
    score_diff: int                 # (player - ai) because MCTS is player
    mcts_final_score: int           # player score
    minimax_final_score: int        # ai score
    mcts_move_times_ms: List[float]
    minimax_move_times_ms: List[float]
    moves_total: int


def generate_coin_line(rng: random.Random, ncoins: int, lo: int, hi: int) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(ncoins)]


def choose_minimax_action(state: cl.State) -> Tuple[Optional[Tuple[str, int]], float]:
    """Return (action, ms_time)."""
    t0 = time.perf_counter()
    # In the provided solution, minimax expects is_maximizing=True on AI turns. :contentReference[oaicite:0]{index=0}
    _, action = cl.minimax(state, is_maximizing=True)
    t1 = time.perf_counter()
    return action, (t1 - t0) * 1000.0


def choose_mcts_action(state: cl.State, budget: int, reward_mode: str, c: float) -> Tuple[Optional[Tuple[str, int]], float]:
    """Return (action, ms_time)."""
    t0 = time.perf_counter()
    action = cl.mcts(state, budget=budget, reward_mode=reward_mode, c=c)
    t1 = time.perf_counter()
    return action, (t1 - t0) * 1000.0


def play_one_game(
    coins: List[int],
    start_turn: str,
    budget: int,
    reward_mode: str,
    c: float,
    max_moves: int = 10_000,
) -> GameResult:
    """
    Plays a single game MCTS('player') vs minimax('ai').
    start_turn: 'player' (MCTS first) or 'ai' (minimax first)
    """
    state = cl.State(coins[:], pScore=0, aiScore=0, turn=start_turn)

    mcts_times: List[float] = []
    mm_times: List[float] = []
    moves = 0

    while not cl.terminal(state):
        moves += 1
        if moves > max_moves:
            raise RuntimeError("Exceeded max_moves; possible bug in succ/terminal/turn switching.")

        turn = cl.player(state)
        if turn == "player":
            action, ms = choose_mcts_action(state, budget=budget, reward_mode=reward_mode, c=c)
            mcts_times.append(ms)
        else:
            action, ms = choose_minimax_action(state)
            mm_times.append(ms)

        if action is None:
            # If an implementation returns None on a non-terminal state, fail loudly.
            raise RuntimeError(f"Got action=None on non-terminal state (turn={turn}, coins_left={len(state.coins)})")

        if action not in cl.actions(state):
            raise RuntimeError(f"Illegal action returned: {action}, legal={cl.actions(state)}")

        state = cl.succ(state, action)

    # Terminal: determine outcome. (MCTS is always the 'player' role.)
    p, a = state.pScore, state.aiScore
    if p > a:
        mcts_w, tie, mm_w = 1, 0, 0
    elif a > p:
        mcts_w, tie, mm_w = 0, 0, 1
    else:
        mcts_w, tie, mm_w = 0, 1, 0

    return GameResult(
        mcts_wins=mcts_w,
        ties=tie,
        minimax_wins=mm_w,
        score_diff=(p - a),
        mcts_final_score=p,
        minimax_final_score=a,
        mcts_move_times_ms=mcts_times,
        minimax_move_times_ms=mm_times,
        moves_total=moves,
    )


@dataclass
class AggregateResult:
    budget: int
    reward_mode: str
    games: int
    win_rate: float
    loss_rate: float
    tie_rate: float
    avg_score_diff: float
    avg_mcts_score: float
    avg_minimax_score: float
    avg_mcts_ms_per_move: float
    avg_minimax_ms_per_move: float


def aggregate(game_results: List[GameResult], budget: int, reward_mode: str) -> AggregateResult:
    games = len(game_results)
    mcts_wins = sum(gr.mcts_wins for gr in game_results)
    minimax_wins = sum(gr.minimax_wins for gr in game_results)
    ties = sum(gr.ties for gr in game_results)

    score_diffs = [gr.score_diff for gr in game_results]
    mcts_scores = [gr.mcts_final_score for gr in game_results]
    mm_scores = [gr.minimax_final_score for gr in game_results]

    mcts_times = [t for gr in game_results for t in gr.mcts_move_times_ms]
    mm_times = [t for gr in game_results for t in gr.minimax_move_times_ms]

    return AggregateResult(
        budget=budget,
        reward_mode=reward_mode,
        games=games,
        win_rate=mcts_wins / games if games else 0.0,
        loss_rate=minimax_wins / games if games else 0.0,
        tie_rate=ties / games if games else 0.0,
        avg_score_diff=stats.mean(score_diffs) if score_diffs else 0.0,
        avg_mcts_score=stats.mean(mcts_scores) if mcts_scores else 0.0,
        avg_minimax_score=stats.mean(mm_scores) if mm_scores else 0.0,
        avg_mcts_ms_per_move=stats.mean(mcts_times) if mcts_times else 0.0,
        avg_minimax_ms_per_move=stats.mean(mm_times) if mm_times else 0.0,
    )


def print_results_table(results: List[AggregateResult]) -> None:
    # Group by reward mode for readability
    results = sorted(results, key=lambda r: (r.reward_mode, r.budget))

    def fmt_pct(x: float) -> str:
        return f"{100.0 * x:5.1f}%"

    def fmt(x: float) -> str:
        return f"{x:8.2f}"

    print()
    for mode in sorted(set(r.reward_mode for r in results)):
        rows = [r for r in results if r.reward_mode == mode]
        print(f"=== Reward mode: {mode} ===")
        print(f"{'Budget':>8}  {'Games':>6}  {'Win%':>7} {'Loss%':>7}  {'Tie%':>7}  {'AvgDiff':>10}  {'MCTS ms/mv':>10}  {'MM ms/mv':>9}")
        for r in rows:
            print(
                f"{r.budget:8d}  {r.games:6d}  {fmt_pct(r.win_rate):>7} {fmt_pct(r.loss_rate):>7}  {fmt_pct(r.tie_rate):>7}  "
                f"{fmt(r.avg_score_diff):>10}  {fmt(r.avg_mcts_ms_per_move):>10}  {fmt(r.avg_minimax_ms_per_move):>9}"
            )
        print()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark MCTS vs minimax for Coins-in-a-Line.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of random coin lines (>=30 recommended).")
    parser.add_argument("--ncoins", type=int, default=DEFAULT_NCOINS, help="Number of coins in the line.")
    parser.add_argument("--minval", type=int, default=1, help="Minimum coin value (inclusive).")
    parser.add_argument("--maxval", type=int, default=10, help="Maximum coin value (inclusive).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--budgets", type=int, nargs="*", default=DEFAULT_BUDGETS, help="MCTS budgets per move.")
    parser.add_argument("--reward_modes", type=str, nargs="*", default=DEFAULT_REWARD_MODES, help="Reward modes to test.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="UCT exploration constant.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Pre-generate coin lines so all budgets/modes see identical test cases
    coin_lines = [generate_coin_line(rng, args.ncoins, args.minval, args.maxval) for _ in range(args.trials)]

    all_results: List[AggregateResult] = []

    for reward_mode in args.reward_modes:
        for budget in args.budgets:
            per_game_results: List[GameResult] = []
            for idx, coins in enumerate(coin_lines):
                # Game A: MCTS starts
                per_game_results.append(
                    play_one_game(coins, start_turn="player", budget=budget, reward_mode=reward_mode, c=args.c)
                )
                # Game B: Minimax starts (same coin line)
                per_game_results.append(
                    play_one_game(coins, start_turn="ai", budget=budget, reward_mode=reward_mode, c=args.c)
                )

                # Lightweight progress indicator
                if (idx + 1) % max(1, args.trials // 5) == 0:
                    print(f"[mode={reward_mode} budget={budget}] completed {idx+1}/{args.trials} coin lines")

            all_results.append(aggregate(per_game_results, budget=budget, reward_mode=reward_mode))

    print_results_table(all_results)

    # Also print CSV for easy copy/paste into spreadsheets
    print("reward_mode,budget,games,win_rate,loss_rate,tie_rate,avg_score_diff,avg_mcts_ms_per_move,avg_minimax_ms_per_move")
    for r in sorted(all_results, key=lambda x: (x.reward_mode, x.budget)):
        print(
            f"{r.reward_mode},{r.budget},{r.games},{r.win_rate:.6f},{r.loss_rate:.6f},{r.tie_rate:.6f},"
            f"{r.avg_score_diff:.6f},{r.avg_mcts_ms_per_move:.6f},{r.avg_minimax_ms_per_move:.6f}"
        )


if __name__ == "__main__":
    main()