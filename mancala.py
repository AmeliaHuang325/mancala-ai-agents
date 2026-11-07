"""
Mancala (Kalah) AI Game Engine
Penn State AI 801 — Group 8 (Matt Kolker, Elliott Armstrong, Shuyang Huang)

A complete Mancala game engine with five AI agents (Random, Greedy,
Minimax, Alpha-Beta, MCTS) for comparative analysis. Supports interactive
play, head-to-head matchups, and automated tournament simulations with
optional CSV data collection.

Agents:
  - Random: uniform random legal move selection
  - Greedy: 1-ply lookahead maximizing own store count
  - Minimax: depth-limited minimax with configurable evaluation function
  - Alpha-Beta: minimax with alpha-beta pruning and move ordering
  - MCTS: Monte Carlo Tree Search with configurable simulation budget and
    exploration constant (C parameter)

Features:
  - Two evaluation functions: standard (store difference) and rich (one-ply
    simulation features including capture/extra-turn analysis)
  - Extra turns do not consume search depth in minimax/alpha-beta
  - Move ordering in alpha-beta: extra-turn moves first, then by store gain
  - MCTS exploration constant (C) is configurable per agent (default sqrt(2))
  - Cross-machine timing normalization via deterministic benchmark
  - CSV output with 40 columns of per-game data for statistical analysis

Board layout:
         <-- P2 pits --
  [12][11][10][ 9][ 8][ 7]
[13]                      [6]
  [ 0][ 1][ 2][ 3][ 4][ 5]
         -- P1 pits -->

Usage:
  python mancala.py                       # Interactive menu
  python mancala.py --tournament          # Default tournament, screen output
  python mancala.py --tournament --csv results.csv   # With CSV output
  python mancala.py --tournament --games 10           # Fewer games (faster)
"""

import csv
import math
import os
import random
import sys
import time
import itertools
from collections import namedtuple
from datetime import datetime

# ---------------------------------------------------------------------------
# AIMA framework classes (adapted from course materials)
# ---------------------------------------------------------------------------

# Game state representation (from AIMA games.py)
GameState = namedtuple('GameState', 'to_move, utility, board, moves')


# Abstract game class (from AIMA games.py)
class Game:
    def actions(self, state):
        raise NotImplementedError

    def result(self, state, move):
        raise NotImplementedError

    def utility(self, state, player):
        raise NotImplementedError

    def terminal_test(self, state):
        return not self.actions(state)

    def to_move(self, state):
        return state.to_move

    def display(self, state):
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    # AIMA's play_game assumes strict alternation; overridden for Mancala extra turns
    def play_game(self, *players):
        state = self.initial
        while True:
            player_fn = players[0] if state.to_move == 1 else players[1]
            move = player_fn(self, state)
            state = self.result(state, move)
            if self.terminal_test(state):
                self.display(state)
                return self.utility(state, 1)


# MCTS tree node (from AIMA utils4e.py)
class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""
    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None


# UCB1 formula (from AIMA utils4e.py; uses math.inf instead of numpy)
def ucb(n, C=1.4):
    return math.inf if n.N == 0 else n.U / n.N + C * math.sqrt(math.log(n.parent.N) / n.N)


# ==========================================================================
# Board constants
# ==========================================================================

P1_PITS = list(range(0, 6))
P1_STORE = 6
P2_PITS = list(range(7, 13))
P2_STORE = 13
OPPOSITE = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7,
            7: 5, 8: 4, 9: 3, 10: 2, 11: 1, 12: 0}


# ==========================================================================
# Mancala game
# ==========================================================================

class Mancala(Game):

    def __init__(self, stones_per_pit=4):
        board = [stones_per_pit] * 14
        board[P1_STORE] = 0
        board[P2_STORE] = 0
        moves = [i for i in P1_PITS if board[i] > 0]
        self.initial = GameState(to_move=1, utility=0, board=tuple(board), moves=tuple(moves))

    def actions(self, state):
        return list(state.moves)

    def result(self, state, move):
        board = list(state.board)
        player = state.to_move
        pits = P1_PITS if player == 1 else P2_PITS
        store = P1_STORE if player == 1 else P2_STORE
        opp_store = P2_STORE if player == 1 else P1_STORE

        if move not in pits or board[move] == 0:
            return state

        stones = board[move]
        board[move] = 0
        idx = move

        while stones > 0:
            idx = (idx + 1) % 14
            if idx == opp_store:
                continue
            board[idx] += 1
            stones -= 1

        if idx in pits and board[idx] == 1 and board[OPPOSITE[idx]] > 0:
            board[store] += board[idx] + board[OPPOSITE[idx]]
            board[idx] = 0
            board[OPPOSITE[idx]] = 0

        next_player = player if idx == store else (2 if player == 1 else 1)

        p1_sum = sum(board[i] for i in P1_PITS)
        p2_sum = sum(board[i] for i in P2_PITS)

        if p1_sum == 0 or p2_sum == 0:
            board[P1_STORE] += p1_sum
            board[P2_STORE] += p2_sum
            for i in P1_PITS + P2_PITS:
                board[i] = 0
            return GameState(to_move=next_player, utility=self._compute_utility(board),
                             board=tuple(board), moves=())

        next_pits = P1_PITS if next_player == 1 else P2_PITS
        next_moves = tuple(i for i in next_pits if board[i] > 0)

        if not next_moves:
            for i in P1_PITS + P2_PITS:
                board[P1_STORE if i in P1_PITS else P2_STORE] += board[i]
                board[i] = 0
            return GameState(to_move=next_player, utility=self._compute_utility(board),
                             board=tuple(board), moves=())

        return GameState(to_move=next_player, utility=0,
                         board=tuple(board), moves=next_moves)

    def _compute_utility(self, board):
        diff = board[P1_STORE] - board[P2_STORE]
        return 1 if diff > 0 else (-1 if diff < 0 else 0)

    def utility(self, state, player):
        return state.utility if player == 1 else -state.utility

    def terminal_test(self, state):
        return len(state.moves) == 0 and (
            sum(state.board[i] for i in P1_PITS) == 0 or
            sum(state.board[i] for i in P2_PITS) == 0 or
            state.utility != 0
        )

    def score(self, state):
        return state.board[P1_STORE], state.board[P2_STORE]

    def eval_fn(self, state, player):
        """Heuristic: normalized store difference. Default evaluation function."""
        return eval_standard(self, state, player)

    def display(self, state):
        b = state.board
        p2_row = '   '.join(f'{b[i]:2d}' for i in reversed(P2_PITS))
        p1_row = '   '.join(f'{b[i]:2d}' for i in P1_PITS)
        print(f'\n     {p2_row}')
        print(f'{b[P2_STORE]:2d}                          {b[P1_STORE]:2d}')
        print(f'     {p1_row}')
        s1, s2 = self.score(state)
        if self.terminal_test(state):
            if s1 > s2:
                status = f'Player 1 wins ({s1}-{s2})'
            elif s2 > s1:
                status = f'Player 2 wins ({s2}-{s1})'
            else:
                status = f'Draw ({s1}-{s2})'
        else:
            status = f'Player {state.to_move} to move'
        print(f'Score: P1={s1}  P2={s2}  |  {status}\n')


# ==========================================================================
# Evaluation functions
#
# Each takes (game, state, player) and returns a score from that player's
# perspective. Higher = better for player. Used by minimax and alpha-beta
# at depth cutoff.
# ==========================================================================

def eval_standard(game, state, player):
    """Standard heuristic: normalized store difference.
    Simple ratio of (my store - opponent store) / total stones.
    Used as the default evaluation function for minimax and alpha-beta.
    """
    s1, s2 = state.board[P1_STORE], state.board[P2_STORE]
    total = sum(state.board)
    if total == 0:
        return 0
    diff = (s1 - s2) / total
    return diff if player == 1 else -diff


def eval_rich(game, state, player):
    """Rich heuristic: store difference + features computed from actual legal moves.

    Instead of static proxies (empty pits, distance-to-store), this evaluates
    each legal move once to count real captures and real extra turns. With at
    most 6 legal moves per side, the cost is minimal.

    Components:
      - Store difference (50%): direct score advantage
      - Side stone advantage (10%): more stones on own side = more options
      - Avg store gain from moves (20%): measures tactical quality of position
      - Extra turn count from moves (20%): actual available free turns
    """
    b = state.board
    s1, s2 = b[P1_STORE], b[P2_STORE]
    total = sum(b)
    if total == 0:
        return 0

    store_diff = (s1 - s2) / total

    p1_side = sum(b[i] for i in P1_PITS)
    p2_side = sum(b[i] for i in P2_PITS)
    side_adv = (p1_side - p2_side) / total

    def move_features(pits, store, to_move):
        """Simulate each legal move, return (avg_store_gain, extra_turn_count)."""
        legal = [i for i in pits if b[i] > 0]
        if not legal:
            return 0.0, 0
        hyp = GameState(to_move=to_move, utility=0, board=state.board, moves=tuple(legal))
        total_gain = 0
        extras = 0
        for move in legal:
            result = game.result(hyp, move)
            total_gain += result.board[store] - b[store]
            if result.to_move == to_move and not game.terminal_test(result):
                extras += 1
        return total_gain / len(legal), extras

    p1_gain, p1_extras = move_features(P1_PITS, P1_STORE, 1)
    p2_gain, p2_extras = move_features(P2_PITS, P2_STORE, 2)

    # Normalize gain by total stones for scale consistency
    gain_adv = (p1_gain - p2_gain) / max(total, 1)
    extra_adv = (p1_extras - p2_extras) / 6.0

    score = (0.50 * store_diff +
             0.10 * side_adv +
             0.20 * gain_adv +
             0.20 * extra_adv)

    return score if player == 1 else -score


# Registry for easy lookup by name
EVAL_FUNCTIONS = {
    'standard': eval_standard,
    'rich': eval_rich,
}


# ==========================================================================
# Players — interface: player(game, state) -> move
# ==========================================================================

def random_player(game, state):
    """A player that chooses a legal move at random."""
    actions = game.actions(state)
    return random.choice(actions) if actions else None


def human_player(game, state):
    game.display(state)
    actions = game.actions(state)
    if not actions:
        return None
    print(f'Player {state.to_move} — choose a pit {actions}: ', end='')
    while True:
        try:
            move = int(input())
            if move in actions:
                return move
            print(f'Invalid. Choose from {actions}: ', end='')
        except (ValueError, EOFError):
            print(f'Enter a number from {actions}: ', end='')


# --- AI Agents ---

def greedy_player(game, state):
    """
    Greedy 1-ply lookahead: Evaluates all legal moves and chooses the one
    that maximizes the current player's store count.
    """
    actions = game.actions(state)
    if not actions:
        return None

    player = state.to_move
    store_idx = P1_STORE if player == 1 else P2_STORE

    best_move = actions[0]
    max_stones = -1

    for move in actions:
        resulting_state = game.result(state, move)
        stones_in_store = resulting_state.board[store_idx]

        if stones_in_store > max_stones:
            max_stones = stones_in_store
            best_move = move
        elif stones_in_store == max_stones:
            if random.random() > 0.5:
                best_move = move

    return best_move

infinity = math.inf

def minimax_player(game, state, depth=4, eval_fn=None):
    """
    Depth-limited minimax player (adapted from AIMA games4e).

    Features:
      - Extra turns do not consume search depth -- the depth counter only
        decrements when the turn passes to the opponent
      - Ties broken randomly to avoid systematic low-index pit bias
      - Configurable evaluation function via eval_fn parameter

    Args:
        depth: maximum search depth (default 4)
        eval_fn: evaluation function (game, state, player) -> score.
                 If None, uses the standard normalized store difference.
    """
    actions = game.actions(state)
    if not actions:
        return None

    root_player = state.to_move

    def evaluate(state):
        if eval_fn is not None:
            return eval_fn(game, state, root_player)
        return game.eval_fn(state, root_player)

    def value(state, depth_left):
        if game.terminal_test(state):
            return game.utility(state, root_player), None
        if depth_left == 0:
            return evaluate(state), None

        if state.to_move == root_player:
            v, move = -infinity, None
            move_list = game.actions(state)
            random.shuffle(move_list)
            for a in move_list:
                next_state = game.result(state, a)
                # Extra turns don't consume depth
                next_depth = depth_left if next_state.to_move == state.to_move else depth_left - 1
                v2, _ = value(next_state, next_depth)
                if v2 > v:
                    v, move = v2, a
            return v, move
        else:
            v, move = +infinity, None
            move_list = game.actions(state)
            random.shuffle(move_list)
            for a in move_list:
                next_state = game.result(state, a)
                next_depth = depth_left if next_state.to_move == state.to_move else depth_left - 1
                v2, _ = value(next_state, next_depth)
                if v2 < v:
                    v, move = v2, a
            return v, move

    best_value, best_move = value(state, depth)
    return best_move


def alpha_beta_player(game, state, depth=6, eval_fn=None):
    """Depth-limited alpha-beta player (adapted from AIMA games4e).

    Features:
      - Alpha-beta pruning for efficient search
      - Move ordering: extra-turn moves first, then by store gain,
        with random tie-breaking to avoid pit-index bias
      - Extra turns do not consume search depth
      - Configurable evaluation function via eval_fn parameter

    Args:
        depth: maximum search depth (default 6)
        eval_fn: evaluation function (game, state, player) -> score.
                 If None, uses the standard normalized store difference.
    """
    actions = game.actions(state)
    if not actions:
        return None

    root_player = state.to_move

    def evaluate(state):
        if eval_fn is not None:
            return eval_fn(game, state, root_player)
        return game.eval_fn(state, root_player)

    def order_moves(state):
        """Order moves: extra turns first, then by immediate store gain.
        Random component breaks ties to avoid systematic bias."""
        player = state.to_move
        store = P1_STORE if player == 1 else P2_STORE
        scored = []
        for a in game.actions(state):
            result = game.result(state, a)
            extra = 100 if result.to_move == player and not game.terminal_test(result) else 0
            gain = result.board[store] - state.board[store]
            scored.append((extra + gain, random.random(), a))
        scored.sort(reverse=True)
        return [a for _, _, a in scored]

    def max_value(state, alpha, beta, depth_left):
        if game.terminal_test(state):
            return game.utility(state, root_player), None
        if depth_left == 0:
            return evaluate(state), None

        v, move = -infinity, None
        for a in order_moves(state):
            next_state = game.result(state, a)

            # Extra turns don't consume depth
            next_depth = depth_left if next_state.to_move == state.to_move else depth_left - 1

            if next_state.to_move == root_player:
                v2, _ = max_value(next_state, alpha, beta, next_depth)
            else:
                v2, _ = min_value(next_state, alpha, beta, next_depth)

            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def min_value(state, alpha, beta, depth_left):
        if game.terminal_test(state):
            return game.utility(state, root_player), None
        if depth_left == 0:
            return evaluate(state), None

        v, move = +infinity, None
        for a in order_moves(state):
            next_state = game.result(state, a)

            next_depth = depth_left if next_state.to_move == state.to_move else depth_left - 1

            if next_state.to_move == root_player:
                v2, _ = max_value(next_state, alpha, beta, next_depth)
            else:
                v2, _ = min_value(next_state, alpha, beta, next_depth)

            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    best_value, best_move = max_value(state, -infinity, +infinity, depth)
    return best_move

def mcts_player(game, state, n_simulations=1000, C=1.4):
    """
    Monte Carlo Tree Search (adapted from AIMA games4e.py).
    Uses MCT_Node and ucb from AIMA utils4e.py.

    Backprop is adapted for Mancala's extra-turn mechanics: the AIMA
    version uses negamax (negate at every level), which assumes strict
    alternation. Mancala allows consecutive turns, so we track utility
    from the root player's perspective and convert at each node.

    Utility mapping: loss(-1)->0, draw(0)->0.5, win(+1)->1, so draws
    are correctly valued between losses and wins.

    C: exploration constant for UCB1 (default 1.4 ~ sqrt(2), per AIMA).
       Lower C (e.g., 0.7-1.0) increases exploitation; higher C increases
       exploration. The optimal value depends on both the simulation
       budget (N) and the opponent.
    """
    root_player = game.to_move(state)
    root = MCT_Node(state=state)

    # Local UCB using the configurable C parameter
    def local_ucb(n):
        return math.inf if n.N == 0 else n.U / n.N + C * math.sqrt(math.log(n.parent.N) / n.N)

    def select(n):
        """Select a leaf node in the tree."""
        if n.children:
            return select(max(n.children.keys(), key=local_ucb))
        else:
            return n

    def expand(n):
        """Expand the leaf node by adding all its children states."""
        if not n.children and not game.terminal_test(n.state):
            actions = game.actions(n.state)
            random.shuffle(actions)  # avoid pit-index bias in initial UCB ties
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in actions}
        return select(n)

    def simulate(sim_state):
        """Simulate a random playout, return utility from root_player's perspective."""
        while not game.terminal_test(sim_state):
            action = random.choice(list(game.actions(sim_state)))
            sim_state = game.result(sim_state, action)
        return game.utility(sim_state, root_player)

    def backprop(n, result):
        """Pass utility back to all parent nodes.
        Maps result from {-1, 0, +1} to {0, 0.5, 1} so draws are valued
        between losses and wins (the standard AIMA backprop only credits
        wins, conflating draws with losses)."""
        n.N += 1
        if n.parent:
            parent_player = game.to_move(n.parent.state)
            u = result if parent_player == root_player else -result
            # Map: loss(-1)=0, draw(0)=0.5, win(+1)=1
            n.U += (u + 1) / 2
        if n.parent:
            backprop(n.parent, result)

    for _ in range(n_simulations):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child.state)
        backprop(child, result)

    if not root.children:
        actions = game.actions(state)
        return actions[0] if actions else None
    max_state = max(root.children, key=lambda p: p.N)
    return root.children.get(max_state)


def make_player(player_fn, **kwargs):
    """Factory to bind keyword args (depth, n_simulations, etc.) into a player function."""
    def player(game, state):
        return player_fn(game, state, **kwargs)
    player.__name__ = f'{player_fn.__name__}({kwargs})'
    return player


# ==========================================================================
# Game runner — per-game data collection
# ==========================================================================

def play_with_stats(p1, p2, stones=4):
    """Run a single game, return per-game statistics dict."""
    game = Mancala(stones_per_pit=stones)
    state = game.initial
    p1_time, p2_time = 0.0, 0.0
    p1_moves, p2_moves = 0, 0
    captures = {1: 0, 2: 0}
    extra_turns = {1: 0, 2: 0}

    while not game.terminal_test(state):
        current = state.to_move
        player_fn = p1 if current == 1 else p2

        t0 = time.perf_counter()
        move = player_fn(game, state)
        elapsed = time.perf_counter() - t0

        if move is None:
            break

        old_store = state.board[P1_STORE if current == 1 else P2_STORE]
        state = game.result(state, move)
        new_store = state.board[P1_STORE if current == 1 else P2_STORE]

        if current == 1:
            p1_time += elapsed
            p1_moves += 1
        else:
            p2_time += elapsed
            p2_moves += 1

        if new_store - old_store > 1:
            captures[current] += 1

        if state.to_move == current and not game.terminal_test(state):
            extra_turns[current] += 1

    s1, s2 = game.score(state)
    winner = 1 if s1 > s2 else (2 if s2 > s1 else 0)

    return {
        'p1_score': s1, 'p2_score': s2, 'winner': winner,
        'score_diff': s1 - s2,
        'total_moves': p1_moves + p2_moves,
        'p1_moves': p1_moves, 'p2_moves': p2_moves,
        'p1_time': p1_time, 'p2_time': p2_time,
        'p1_avg_time': p1_time / max(p1_moves, 1),
        'p2_avg_time': p2_time / max(p2_moves, 1),
        'p1_captures': captures[1], 'p2_captures': captures[2],
        'p1_extra_turns': extra_turns[1], 'p2_extra_turns': extra_turns[2],
        'first_mover_won': winner == 1,
        'first_mover_score': s1, 'second_mover_score': s2,
    }


# ==========================================================================
# Agent metadata — parse labels into structured fields for CSV
# ==========================================================================

def parse_agent_label(label):
    """Parse a label like 'Alpha-Beta(d=4,rich)' or 'MCTS(n=5000,C=0.5)'.
    Returns: (agent_type, param_name, param_value, eval_fn_name, explore_c)
    """
    eval_fn_name = 'standard'
    explore_c = ''

    if '(' not in label:
        return label.lower(), '', '', eval_fn_name, explore_c

    name_part = label[:label.index('(')]
    param_part = label[label.index('(') + 1:label.index(')')]
    agent_type = name_part.strip().lower().replace('-', '_')

    param_name, param_value = '', ''
    parts = [p.strip() for p in param_part.split(',')]

    for part in parts:
        if '=' in part:
            k, v = part.split('=', 1)
            k, v = k.strip(), v.strip()
            if k in ('d', 'n'):
                param_name, param_value = k, v
            elif k == 'C':
                explore_c = v
        elif part in EVAL_FUNCTIONS:
            eval_fn_name = part

    # Default explore_c for MCTS agents
    if agent_type == 'mcts' and explore_c == '':
        explore_c = '1.4'

    return agent_type, param_name, param_value, eval_fn_name, explore_c


# ==========================================================================
# CSV output
# ==========================================================================

CSV_COLUMNS = [
    'simulation_id', 'scenario_name', 'matchup_label', 'game_num',
    'p1_label', 'p2_label',
    'p1_agent_type', 'p1_param_name', 'p1_param_value',
    'p1_eval_fn', 'p1_explore_c',
    'p2_agent_type', 'p2_param_name', 'p2_param_value',
    'p2_eval_fn', 'p2_explore_c',
    'p1_went_first',
    'p1_score', 'p2_score', 'score_diff',
    'winner',
    'p1_moves', 'p2_moves', 'total_moves',
    'p1_time', 'p2_time', 'p1_avg_time', 'p2_avg_time',
    'p1_captures', 'p2_captures',
    'p1_extra_turns', 'p2_extra_turns',
    'first_mover_won', 'first_mover_label',
    'first_mover_score', 'second_mover_score',
    'stones_per_pit', 'machine_id', 'benchmark_ms', 'timestamp',
]

# Module-level globals set once at startup by run_benchmark()
MACHINE_ID = 'default'
BENCHMARK_MS = 0.0


def run_benchmark(n_runs=10):
    """Benchmark: run AB(d=6) on initial state n_runs times, return avg ms.
    Deterministic search on a fixed position — measures raw CPU/Python speed.
    Sets module globals MACHINE_ID and BENCHMARK_MS."""
    global BENCHMARK_MS
    game = Mancala()
    state = game.initial
    # Warm up
    alpha_beta_player(game, state, depth=6)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        alpha_beta_player(game, state, depth=6)
        times.append((time.perf_counter() - t0) * 1000)
    BENCHMARK_MS = sum(times) / len(times)
    return BENCHMARK_MS


def init_csv(filepath):
    """Create CSV with header if file does not exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def write_game_to_csv(filepath, row_dict):
    """Append a single game row to the CSV."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(col, '') for col in CSV_COLUMNS])


def build_csv_row(game_stats, p1_label, p2_label, p1_went_first,
                  game_num, simulation_id, scenario_name, matchup_label, stones=4):
    """Convert a play_with_stats result dict into a CSV row dict."""
    p1_type, p1_pname, p1_pval, p1_eval, p1_c = parse_agent_label(p1_label)
    p2_type, p2_pname, p2_pval, p2_eval, p2_c = parse_agent_label(p2_label)
    g = game_stats

    winner_str = 'p1' if g['winner'] == 1 else ('p2' if g['winner'] == 2 else 'draw')
    first_mover_label = p1_label if p1_went_first else p2_label

    return {
        'simulation_id': simulation_id,
        'scenario_name': scenario_name,
        'matchup_label': matchup_label,
        'game_num': game_num,
        'p1_label': p1_label,
        'p2_label': p2_label,
        'p1_agent_type': p1_type,
        'p1_param_name': p1_pname,
        'p1_param_value': p1_pval,
        'p1_eval_fn': p1_eval,
        'p1_explore_c': p1_c,
        'p2_agent_type': p2_type,
        'p2_param_name': p2_pname,
        'p2_param_value': p2_pval,
        'p2_eval_fn': p2_eval,
        'p2_explore_c': p2_c,
        'p1_went_first': p1_went_first,
        'p1_score': g['p1_score'],
        'p2_score': g['p2_score'],
        'score_diff': g['score_diff'],
        'winner': winner_str,
        'p1_moves': g['p1_moves'],
        'p2_moves': g['p2_moves'],
        'total_moves': g['total_moves'],
        'p1_time': round(g['p1_time'], 6),
        'p2_time': round(g['p2_time'], 6),
        'p1_avg_time': round(g['p1_avg_time'], 6),
        'p2_avg_time': round(g['p2_avg_time'], 6),
        'p1_captures': g['p1_captures'],
        'p2_captures': g['p2_captures'],
        'p1_extra_turns': g['p1_extra_turns'],
        'p2_extra_turns': g['p2_extra_turns'],
        'first_mover_won': g['first_mover_won'],
        'first_mover_label': first_mover_label,
        'first_mover_score': g['first_mover_score'],
        'second_mover_score': g['second_mover_score'],
        'stones_per_pit': stones,
        'machine_id': MACHINE_ID,
        'benchmark_ms': round(BENCHMARK_MS, 3),
        'timestamp': datetime.now().isoformat(),
    }



# ==========================================================================
# Matchup runner
# ==========================================================================

def run_matchup(p1_fn, p2_fn, p1_label, p2_label, n_games=25,
                csv_path=None, simulation_id='', scenario_name='',
                stones=4, show_progress=True):
    """Run n_games with p1 first, then n_games with p2 first (2*n_games total).
    If csv_path is provided, writes each game to CSV. Returns summary dict.
    """
    matchup_label = f'{p1_label} vs {p2_label}'
    total = 2 * n_games
    p1_wins, p2_wins, draws = 0, 0, 0
    p1_total_time, p2_total_time = 0.0, 0.0
    total_moves_sum = 0
    all_games = []

    for i in range(total):
        if show_progress:
            print(f'\r  {matchup_label}: game {i+1}/{total}...', end='', flush=True)

        if i < n_games:
            g = play_with_stats(p1_fn, p2_fn, stones=stones)
            p1_went_first = True
        else:
            raw = play_with_stats(p2_fn, p1_fn, stones=stones)
            g = {
                'p1_score': raw['p2_score'], 'p2_score': raw['p1_score'],
                'winner': (2 if raw['winner'] == 1 else (1 if raw['winner'] == 2 else 0)),
                'score_diff': -(raw['score_diff']),
                'total_moves': raw['total_moves'],
                'p1_moves': raw['p2_moves'], 'p2_moves': raw['p1_moves'],
                'p1_time': raw['p2_time'], 'p2_time': raw['p1_time'],
                'p1_avg_time': raw['p2_avg_time'], 'p2_avg_time': raw['p1_avg_time'],
                'p1_captures': raw['p2_captures'], 'p2_captures': raw['p1_captures'],
                'p1_extra_turns': raw['p2_extra_turns'], 'p2_extra_turns': raw['p1_extra_turns'],
                'first_mover_won': raw['first_mover_won'],
                'first_mover_score': raw['first_mover_score'],
                'second_mover_score': raw['second_mover_score'],
            }
            p1_went_first = False

        if g['winner'] == 1: p1_wins += 1
        elif g['winner'] == 2: p2_wins += 1
        else: draws += 1
        p1_total_time += g['p1_time']
        p2_total_time += g['p2_time']
        total_moves_sum += g['total_moves']

        # Write to CSV only if path provided
        if csv_path:
            row = build_csv_row(g, p1_label, p2_label, p1_went_first,
                                i + 1, simulation_id, scenario_name, matchup_label, stones)
            write_game_to_csv(csv_path, row)
        all_games.append(g)

    p1_avg_score = sum(g['p1_score'] for g in all_games) / total
    p2_avg_score = sum(g['p2_score'] for g in all_games) / total
    avg_moves = total_moves_sum / total

    summary = {
        'matchup_label': matchup_label,
        'total_games': total, 'p1_wins': p1_wins, 'p2_wins': p2_wins, 'draws': draws,
        'p1_avg_score': p1_avg_score, 'p2_avg_score': p2_avg_score,
        'avg_moves': avg_moves,
        'p1_total_time': p1_total_time, 'p2_total_time': p2_total_time,
    }

    if show_progress:
        elapsed = p1_total_time + p2_total_time
        print(f'\r  {matchup_label:<45s} '
              f'{p1_wins:>2d}W-{p2_wins:>2d}L-{draws:>2d}D  '
              f'avg={p1_avg_score:.1f}-{p2_avg_score:.1f}  '
              f'mvs={avg_moves:.0f}  '
              f'({elapsed:.1f}s)          ')

    return summary


def run_scenario(matchups, n_games=25, csv_path=None,
                 scenario_name='', stones=4):
    """Run a list of matchups.
    matchups: list of (p1_label, p1_fn, p2_label, p2_fn) tuples
    csv_path: if provided, writes game-level data to CSV; if None, screen only
    """
    simulation_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_matchups = len(matchups)
    total_games_planned = total_matchups * 2 * n_games

    print(f'\n{"=" * 80}')
    print(f'SCENARIO: {scenario_name}')
    print(f'{"=" * 80}')
    print(f'Matchups: {total_matchups}')
    print(f'Games per matchup: {2 * n_games} ({n_games} each as P1)')
    print(f'Total games: {total_games_planned}')
    if csv_path:
        print(f'CSV output: {csv_path}')
    else:
        print(f'Output: screen only (no CSV)')
    print(f'Simulation ID: {simulation_id}')
    print()

    if csv_path:
        init_csv(csv_path)
    summaries = []
    scenario_t0 = time.time()

    for idx, (p1_label, p1_fn, p2_label, p2_fn) in enumerate(matchups):
        s = run_matchup(p1_fn, p2_fn, p1_label, p2_label,
                        n_games=n_games, csv_path=csv_path,
                        simulation_id=simulation_id,
                        scenario_name=scenario_name, stones=stones)
        summaries.append(s)

    scenario_elapsed = time.time() - scenario_t0
    print(f'\nScenario complete: {total_matchups} matchups, '
          f'{total_games_planned} games in {scenario_elapsed:.1f}s')
    print()

    return summaries


# ==========================================================================
# Agent catalog
#
# All available agents with display labels. To add a new agent:
#   1. Add an entry here: 'Label': ('Label', make_player(fn, **kwargs))
#   2. Reference it in build_tournament_matchups() below
#
# The label is used in CSV output, matchup display, and results tables.
# For MCTS, the C parameter defaults to 1.4 (sqrt(2)) unless overridden.
# For Alpha-Beta, eval_fn defaults to eval_standard unless overridden.
# ==========================================================================

AGENT_CATALOG = {
    # --- Baseline agents ---
    'Random':             ('Random', random_player),
    'Greedy':             ('Greedy', greedy_player),

    # --- Minimax (depth-limited, no pruning) ---
    'Minimax(d=4)':       ('Minimax(d=4)', make_player(minimax_player, depth=4)),
    'Minimax(d=6)':       ('Minimax(d=6)', make_player(minimax_player, depth=6)),

    # --- Alpha-Beta (depth-limited with pruning + move ordering) ---
    'Alpha-Beta(d=4)':    ('Alpha-Beta(d=4)', make_player(alpha_beta_player, depth=4)),
    'Alpha-Beta(d=6)':    ('Alpha-Beta(d=6)', make_player(alpha_beta_player, depth=6)),
    'Alpha-Beta(d=8)':    ('Alpha-Beta(d=8)', make_player(alpha_beta_player, depth=8)),

    # --- MCTS (simulation-based, default C=1.4) ---
    'MCTS(n=1000)':       ('MCTS(n=1000)', make_player(mcts_player, n_simulations=1000)),
    'MCTS(n=2000)':       ('MCTS(n=2000)', make_player(mcts_player, n_simulations=2000)),
    'MCTS(n=5000)':       ('MCTS(n=5000)', make_player(mcts_player, n_simulations=5000)),
    'MCTS(n=10000)':      ('MCTS(n=10000)', make_player(mcts_player, n_simulations=10000)),
    'MCTS(n=20000)':      ('MCTS(n=20000)', make_player(mcts_player, n_simulations=20000)),
    'MCTS(n=50000)':      ('MCTS(n=50000)', make_player(mcts_player, n_simulations=50000)),

    # --- Rich evaluation function variants ---
    # Uses eval_rich (one-ply simulation features) instead of eval_standard
    'Alpha-Beta(d=4,rich)': ('Alpha-Beta(d=4,rich)', make_player(alpha_beta_player, depth=4, eval_fn=eval_rich)),
    'Alpha-Beta(d=6,rich)': ('Alpha-Beta(d=6,rich)', make_player(alpha_beta_player, depth=6, eval_fn=eval_rich)),

    # --- MCTS C parameter variants ---
    # To test different exploration constants, add entries like:
    # 'MCTS(n=5000,C=0.7)': ('MCTS(n=5000,C=0.7)', make_player(mcts_player, n_simulations=5000, C=0.7)),
    # 'MCTS(n=5000,C=1.0)': ('MCTS(n=5000,C=1.0)', make_player(mcts_player, n_simulations=5000, C=1.0)),
}


# ==========================================================================
# Tournament mode — default matchup definitions
#
# HOW TO CUSTOMIZE:
#   - Adjust N_GAMES to change games per matchup (total = 2 * N_GAMES)
#   - Add/remove matchups in build_tournament_matchups() below
#   - Add agents to AGENT_CATALOG above, then reference them here
#   - Each matchup plays N_GAMES with agent A going first, then N_GAMES
#     with agent B going first, for balanced first-mover analysis
#   - Uncomment the EXTENDED sections for deeper (but slower) analysis
# ==========================================================================

N_GAMES = 25  # Games per seating position. Total per matchup = 2 * N_GAMES.

def build_tournament_matchups():
    """Build the default tournament matchup list.
    Returns list of (scenario_name, matchup_list) tuples.

    Default tournament (~28 matchups, ~1400 games, ~2-3 hours):
      1. Calibration: every core agent vs Greedy (establishes hierarchy)
      2. AB depth scaling: d=4 vs d=6 vs d=8 (depth benefit analysis)
      3. MM vs AB: matched-depth comparison (pruning speedup measurement)
      4. MCTS vs AB: simulation vs search crossover analysis
      5. MCTS internal scaling: N vs N (budget benefit analysis)
      6. Self-play: first-mover advantage analysis
      7. Random baseline: floor calibration

    To add matchups: use the matchup() and self_play() helpers below.
    To run expensive agents (n=20k+), uncomment the EXTENDED sections.
    """
    scenarios = []

    def agent(name):
        return AGENT_CATALOG[name]

    def matchup(a_name, b_name):
        a_label, a_fn = agent(a_name)
        b_label, b_fn = agent(b_name)
        return (a_label, a_fn, b_label, b_fn)

    def self_play(name):
        label, fn = agent(name)
        return (label, fn, label, fn)

    # --- 1. Calibration vs Greedy ---
    # Establishes the hierarchy. Every core agent plays Greedy.
    cal = [self_play('Greedy')]
    for name in ['Minimax(d=4)', 'Minimax(d=6)',
                  'Alpha-Beta(d=4)', 'Alpha-Beta(d=6)', 'Alpha-Beta(d=8)',
                  'MCTS(n=1000)', 'MCTS(n=5000)', 'MCTS(n=10000)']:
        cal.append(matchup(name, 'Greedy'))
    scenarios.append(('Calibration vs Greedy', cal))

    # --- 2. AB depth scaling ---
    # How much does each +2 depth buy?
    scenarios.append(('AB depth scaling', [
        matchup('Alpha-Beta(d=4)', 'Alpha-Beta(d=6)'),
        matchup('Alpha-Beta(d=6)', 'Alpha-Beta(d=8)'),
        matchup('Alpha-Beta(d=4)', 'Alpha-Beta(d=8)'),
    ]))

    # --- 3. MM vs AB at matched depth ---
    # Confirm identical play quality, measure pruning speedup
    scenarios.append(('MM vs AB (same depth)', [
        matchup('Minimax(d=4)', 'Alpha-Beta(d=4)'),
        matchup('Minimax(d=6)', 'Alpha-Beta(d=6)'),
    ]))

    # --- 4. MCTS vs Alpha-Beta ---
    # The core comparison: where does simulation-based search
    # break even with depth-limited adversarial search?
    scenarios.append(('MCTS vs Alpha-Beta', [
        matchup('MCTS(n=1000)',  'Alpha-Beta(d=4)'),
        matchup('MCTS(n=5000)',  'Alpha-Beta(d=4)'),
        matchup('MCTS(n=10000)', 'Alpha-Beta(d=4)'),
        matchup('MCTS(n=5000)',  'Alpha-Beta(d=6)'),
        matchup('MCTS(n=10000)', 'Alpha-Beta(d=6)'),
        matchup('MCTS(n=10000)', 'Alpha-Beta(d=8)'),
    ]))

    # --- 5. MCTS internal scaling ---
    # How much does increasing N help in direct MCTS vs MCTS play?
    scenarios.append(('MCTS internal scaling', [
        matchup('MCTS(n=1000)',  'MCTS(n=5000)'),
        matchup('MCTS(n=5000)',  'MCTS(n=10000)'),
        matchup('MCTS(n=1000)',  'MCTS(n=10000)'),
    ]))

    # --- 6. Self-play (first-mover analysis) ---
    scenarios.append(('Self-play', [
        self_play('Alpha-Beta(d=4)'),
        self_play('Alpha-Beta(d=6)'),
        self_play('Alpha-Beta(d=8)'),
        self_play('MCTS(n=5000)'),
    ]))

    # --- 7. Random baseline ---
    # One matchup establishes the floor.
    scenarios.append(('Random baseline', [
        matchup('Alpha-Beta(d=4)', 'Random'),
    ]))

    # =====================================================================
    # EXTENDED MATCHUPS — uncomment for deeper analysis (adds hours)
    # =====================================================================

    # # --- MCTS high-N vs AB (~4+ hours) ---
    # # Tests whether MCTS with large budgets can compete with deep AB.
    # scenarios.append(('MCTS high-N', [
    #     matchup('MCTS(n=20000)', 'Alpha-Beta(d=4)'),
    #     matchup('MCTS(n=20000)', 'Alpha-Beta(d=6)'),
    #     matchup('MCTS(n=20000)', 'Alpha-Beta(d=8)'),
    #     matchup('MCTS(n=50000)', 'Alpha-Beta(d=4)'),
    #     matchup('MCTS(n=50000)', 'Alpha-Beta(d=6)'),
    # ]))

    # # --- Rich eval function comparison ---
    # # Compares the standard eval (store difference) against the rich eval
    # # (one-ply simulation features) at matched search depths.
    # scenarios.append(('Eval: standard vs rich', [
    #     matchup('Alpha-Beta(d=4)', 'Alpha-Beta(d=4,rich)'),
    #     matchup('Alpha-Beta(d=6)', 'Alpha-Beta(d=6,rich)'),
    # ]))

    # # --- MCTS C parameter sweep ---
    # # Tests different exploration constants. Add C variants to AGENT_CATALOG:
    # #   'MCTS(n=5000,C=0.7)': ('MCTS(n=5000,C=0.7)', make_player(mcts_player, n_simulations=5000, C=0.7)),
    # #   'MCTS(n=5000,C=1.0)': ('MCTS(n=5000,C=1.0)', make_player(mcts_player, n_simulations=5000, C=1.0)),
    # # Then uncomment:
    # # scenarios.append(('C parameter sweep', [
    # #     matchup('MCTS(n=5000,C=0.7)', 'Alpha-Beta(d=4)'),
    # #     matchup('MCTS(n=5000,C=1.0)', 'Alpha-Beta(d=4)'),
    # # ]))

    # # --- Complete Random baselines ---
    # # Every agent vs Random for a full floor calibration.
    # scenarios.append(('All vs Random', [
    #     matchup('Random', 'Greedy'),
    #     matchup('Minimax(d=4)', 'Random'),
    #     matchup('Alpha-Beta(d=6)', 'Random'),
    #     matchup('MCTS(n=1000)', 'Random'),
    #     matchup('MCTS(n=5000)', 'Random'),
    # ]))

    return scenarios


# ==========================================================================
# Interactive single-game mode
# ==========================================================================

def play_interactive():
    """Play a single game interactively: choose agents, watch or play."""
    agents_list = list(AGENT_CATALOG.keys()) + ['Human']

    print('\n  Available agents:')
    for i, name in enumerate(agents_list):
        print(f'    {i+1:>2}. {name}')

    def pick_agent(prompt):
        while True:
            try:
                choice = int(input(prompt)) - 1
                if 0 <= choice < len(agents_list):
                    name = agents_list[choice]
                    if name == 'Human':
                        return 'Human', human_player
                    return AGENT_CATALOG[name]
                print(f'  Choose 1-{len(agents_list)}')
            except (ValueError, EOFError):
                print(f'  Enter a number 1-{len(agents_list)}')

    p1_label, p1_fn = pick_agent('\n  Player 1: ')
    p2_label, p2_fn = pick_agent('  Player 2: ')

    print(f'\n  {p1_label} vs {p2_label}\n')

    game = Mancala()
    state = game.initial

    p1_time, p2_time = 0.0, 0.0
    move_count = 0

    while not game.terminal_test(state):
        current = state.to_move
        fn = p1_fn if current == 1 else p2_fn
        label = p1_label if current == 1 else p2_label

        if fn != human_player:
            game.display(state)

        t0 = time.perf_counter()
        move = fn(game, state)
        elapsed = time.perf_counter() - t0

        if current == 1:
            p1_time += elapsed
        else:
            p2_time += elapsed
        move_count += 1

        if fn != human_player:
            print(f'  {label} plays pit {move} ({elapsed*1000:.0f}ms)')

        state = game.result(state, move)

    game.display(state)
    s1, s2 = game.score(state)
    print(f'  P1 time: {p1_time:.2f}s  |  P2 time: {p2_time:.2f}s  |  {move_count} moves')


# ==========================================================================
# Main — interactive menu
# ==========================================================================

def print_summary_table(summaries):
    """Print a formatted results table from a list of run_matchup summaries."""
    print(f'\n{"=" * 80}')
    print(f'{"Matchup":<45s} {"N":>4} {"P1W":>4} {"P2W":>4} {"D":>3} {"P1%":>5} {"AvgSc":>11} {"Mvs":>4}')
    print(f'{"-" * 80}')
    for s in summaries:
        wr = s['p1_wins'] / s['total_games'] * 100
        print(f'{s["matchup_label"]:<45s} {s["total_games"]:>4} '
              f'{s["p1_wins"]:>4} {s["p2_wins"]:>4} {s["draws"]:>3} '
              f'{wr:>4.0f}% '
              f'{s["p1_avg_score"]:>5.1f}-{s["p2_avg_score"]:<5.1f} '
              f'{s["avg_moves"]:>4.0f}')
    print(f'{"=" * 80}\n')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Mancala (Kalah) AI Engine — Play, Watch, or Run Tournaments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mancala.py                       # Interactive menu
  python mancala.py --tournament          # Run default tournament, results to screen
  python mancala.py --tournament --csv results.csv     # Tournament with CSV output
  python mancala.py --tournament --csv results.csv --machine laptop_A
  python mancala.py --tournament --games 10            # Fewer games (faster)
        """)
    parser.add_argument('--tournament', action='store_true',
                        help='Run default tournament (skip interactive menu)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Write game-level results to this CSV file. '
                             'If omitted, results shown on screen only.')
    parser.add_argument('--machine', type=str, default='default',
                        help='Machine ID for cross-machine timing normalization')
    parser.add_argument('--games', type=int, default=25,
                        help='Games per seating (total = 2x this). Default: 25')
    args = parser.parse_args()

    global MACHINE_ID, N_GAMES
    MACHINE_ID = args.machine
    N_GAMES = args.games

    # Run benchmark for timing normalization
    print(f'Machine: {MACHINE_ID}')
    print('Running benchmark...')
    bm = run_benchmark(10)
    print(f'Benchmark: {bm:.1f} ms (AB d=6 on initial state)\n')

    if args.tournament:
        # ---- Tournament mode ----
        scenarios = build_tournament_matchups()
        total_matchups = sum(len(m) for _, m in scenarios)
        total_games = total_matchups * 2 * N_GAMES

        print('=' * 80)
        print('MANCALA AI TOURNAMENT')
        print('=' * 80)
        print(f'Machine: {MACHINE_ID}, Benchmark: {bm:.1f} ms')
        print(f'Scenarios: {len(scenarios)}, Matchups: {total_matchups}, Games: {total_games}')
        if args.csv:
            print(f'CSV: {args.csv}')
        else:
            print('Output: screen only')
        print()
        for name, matchups in scenarios:
            print(f'  {name} ({len(matchups)} matchups):')
            for m in matchups:
                print(f'    {m[0]} vs {m[2]}')
        print()

        overall_t0 = time.time()
        all_summaries = []
        for sn, matchups in scenarios:
            summaries = run_scenario(matchups, n_games=N_GAMES,
                                     csv_path=args.csv, scenario_name=sn)
            all_summaries.extend(summaries)

        elapsed = time.time() - overall_t0
        print_summary_table(all_summaries)
        print(f'Tournament complete: {elapsed:.0f}s ({elapsed/60:.1f} min)')
        if args.csv:
            print(f'Results saved to: {args.csv}')
        return

    # ---- Interactive menu ----
    while True:
        print('\n' + '=' * 50)
        print('  MANCALA AI ENGINE')
        print('=' * 50)
        print('  1. Play / Watch a single game')
        print('  2. Run a quick matchup (2 agents, N games)')
        print('  3. Run full tournament')
        print('  4. Quit')
        print()

        try:
            choice = input('  Choice: ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == '1':
            play_interactive()

        elif choice == '2':
            agents_list = list(AGENT_CATALOG.keys())
            print('\n  Agents:')
            for i, name in enumerate(agents_list):
                print(f'    {i+1:>2}. {name}')

            try:
                a = int(input('\n  Agent A: ')) - 1
                b = int(input('  Agent B: ')) - 1
                n = int(input('  Games per seating (total = 2x): '))
                a_label, a_fn = AGENT_CATALOG[agents_list[a]]
                b_label, b_fn = AGENT_CATALOG[agents_list[b]]
            except (ValueError, IndexError, EOFError):
                print('  Invalid input.')
                continue

            csv_q = input(f'  Save to CSV? (enter filename or press Enter to skip): ').strip()
            csv_out = csv_q if csv_q else None

            s = run_matchup(a_fn, b_fn, a_label, b_label,
                            n_games=n, csv_path=csv_out,
                            simulation_id=datetime.now().strftime('%Y%m%d_%H%M%S'),
                            scenario_name='Quick matchup')
            print_summary_table([s])

        elif choice == '3':
            csv_q = input('  Save to CSV? (enter filename or press Enter to skip): ').strip()
            csv_out = csv_q if csv_q else None

            scenarios = build_tournament_matchups()
            overall_t0 = time.time()
            all_summaries = []
            for sn, matchups in scenarios:
                summaries = run_scenario(matchups, n_games=N_GAMES,
                                         csv_path=csv_out, scenario_name=sn)
                all_summaries.extend(summaries)

            elapsed = time.time() - overall_t0
            print_summary_table(all_summaries)
            print(f'  Tournament complete: {elapsed:.0f}s ({elapsed/60:.1f} min)')
            if csv_out:
                print(f'  Results saved to: {csv_out}')

        elif choice == '4':
            break

        else:
            print('  Choose 1-4.')


if __name__ == '__main__':
    main()
