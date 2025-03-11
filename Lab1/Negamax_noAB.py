"""
The standard AI algorithm of easyAI is Negamax with alpha-beta pruning
and (optionnally), transposition tables.
"""

import pickle

LOWERBOUND, EXACT, UPPERBOUND = -1, 0, 1
inf = float("infinity")

# Removed the original negamax() function that uses alpha-beta pruning
# Added a new pure negamax function without alpha-beta pruning
def negamax_no_ab(game, depth, scoring):
    if depth == 0 or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)
    best_value = -inf
    for move in game.possible_moves():
        game.make_move(move)
        game.switch_player()
        value = -negamax_no_ab(game, depth - 1, scoring)
        game.switch_player()
        game.unmake_move(move)
        best_value = max(best_value, value)
    return best_value

class NegamaxNoAB:
    """
    Modified Negamax algorithm without alpha-beta pruning.
    """
    def __init__(self, depth, scoring=None, win_score=+inf):
        self.scoring = scoring
        self.depth = depth
        self.win_score = win_score

    def __call__(self, game):
        scoring = self.scoring if self.scoring else (lambda g: g.scoring())
        best_value = -inf
        best_move = None
        for move in game.possible_moves():
            game.make_move(move)
            game.switch_player()
            value = -negamax_no_ab(game, self.depth - 1, scoring)
            game.switch_player()
            game.unmake_move(move)
            if value > best_value:
                best_value = value
                best_move = move
        game.ai_move = best_move
        return best_move
