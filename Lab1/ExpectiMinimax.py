from math import inf

LOWERBOUND, EXACT, UPPERBOUND = -1, 0, 1

def star_minimax(game, depth, origDepth, scoring, alpha, beta, tt=None):
    """
    A simplified implementation of *-minimax (expectiminimax) with partial alpha-beta.
    
    Pseudocode references:
      1) If node is terminal or depth = 0 => return scoring
      2) If node is a max/min node => do normal minimax 
      3) Otherwise => do chance node (the *-minimax part)
    
    The chance node portion follows your pseudocode:
        N = numSuccessors(node)
        A = N*(alpha - U) + U
        B = N*(beta  - L) + L
        sum = 0
        foreach child:
            AX = max(A, L)
            BX = min(B, U)
            score = star_minimax(child, ...)
            if score <= A: return alpha
            if score >= B: return beta
            sum += score
            A += (U - score)
            B += (L - score)
        return sum / N

    We store the best move in game.ai_move when at the top-level (depth == origDepth).
    We also optionally use a transposition table (tt) exactly as done in easyAI's negamax.
    """

    # 1) Check if we have a transposition-table (TT) and entry for this game
    lookup = None if (tt is None) else tt.lookup(game)
    alphaOrig = alpha

    if lookup is not None:
        if lookup["depth"] >= depth:
            # We can possibly reuse stored data
            flag, value = lookup["flag"], lookup["value"]
            if flag == EXACT:
                if depth == origDepth:
                    game.ai_move = lookup["move"]
                return value
            elif flag == LOWERBOUND:
                alpha = max(alpha, value)
            elif flag == UPPERBOUND:
                beta = min(beta, value)
            if alpha >= beta:
                if depth == origDepth:
                    game.ai_move = lookup["move"]
                return value

    # 2) Terminal or depth limit => scoring
    if (depth == 0) or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)

    # 3) Decide if this is a max/min node or a chance node
    #    (This logic depends on your game. For standard two-player zero-sum, we
    #     might treat both players with a 'negamax' approach => max node if I'm the
    #     current_player, min node if I'm the opponent, etc. Or you might do
    #     if game.is_chance_node(): chance code else: minimax code.)
    #    
    #    For illustration, let's do:
    #       - If game.probabilistic == False => treat as a 'max node' under negamax
    #       - Else => treat as a chance node
    #    Adjust to your needs.

    if not getattr(game, "probabilistic", False):
        # ---- This is a standard "max node" approach (negamax style) ----
        possible_moves = game.possible_moves()
        if lookup is not None and (lookup["move"] in possible_moves):
            # Put stored best move first
            bm = lookup["move"]
            possible_moves.remove(bm)
            possible_moves = [bm] + possible_moves
        else:
            possible_moves = game.possible_moves()

        state = game
        best_move = possible_moves[0]
        if depth == origDepth:
            state.ai_move = best_move

        best_value = -inf
        unmake_move = hasattr(state, "unmake_move")

        for move in possible_moves:
            if not unmake_move:
                child = state.copy()
                child.make_move(move)
                child.switch_player()
                score = -star_minimax(child, depth - 1, origDepth, scoring, -beta, -alpha, tt)
            else:
                # do in-place
                state.make_move(move)
                state.switch_player()
                score = -star_minimax(state, depth - 1, origDepth, scoring, -beta, -alpha, tt)
                state.switch_player()
                state.unmake_move(move)

            if score > best_value:
                best_value = score
                best_move = move

            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        if depth == origDepth:
            game.ai_move = best_move

        # Possibly store in TT
        if tt is not None:
            flag = (
                UPPERBOUND if (best_value <= alphaOrig) 
                else (LOWERBOUND if (best_value >= beta) else EXACT)
            )
            tt.store(game=state, depth=depth, value=best_value, move=best_move, flag=flag)

        return best_value

    else:
        # ---- This is the *chance node* portion (your pseudocode) ----
        possible_moves = game.possible_moves()
        N = len(possible_moves)
        if N == 0:
            # No successors => treat as terminal
            return scoring(game) * (1 + 0.001 * depth)

        # We define L = -∞, U = +∞ as in the pseudocode
        L, U = -inf, +inf
        A = N*(alpha - U) + U   # = N*alpha - N*U + U
        B = N*(beta  - L) + L   # = N*beta  - N*L + L

        # If TT gave us a recommended move, put it first
        if lookup is not None and (lookup["move"] in possible_moves):
            bm = lookup["move"]
            possible_moves.remove(bm)
            possible_moves = [bm] + possible_moves

        total_score = 0.0
        unmake_move = hasattr(game, "unmake_move")

        # We won't necessarily track "best move" in a chance node, because
        # there's no single "best" outcome if all moves happen randomly.
        # But if you have a reason to pick a 'representative' move, you can.
        # For consistency, let's track the first child that doesn't prune.
        best_move = possible_moves[0]
        pruned_early = False

        for i, move in enumerate(possible_moves):
            # Evaluate child
            if not unmake_move:
                child = game.copy()
                child.make_move(move)
                child.switch_player()
                score = star_minimax(child, depth - 1, origDepth, scoring, 
                                     max(A, L), min(B, U), tt)
            else:
                game.make_move(move)
                game.switch_player()
                score = star_minimax(game, depth - 1, origDepth, scoring, 
                                     max(A, L), min(B, U), tt)
                game.switch_player()
                game.unmake_move(move)

            if i == 0:
                best_move = move  # so we have something to store if needed

            # Check cutoffs
            if score <= A:
                # cutoff => return alpha
                pruned_early = True
                total_score = alpha  # interpret as immediate alpha cutoff
                break
            if score >= B:
                # cutoff => return beta
                pruned_early = True
                total_score = beta   # interpret as immediate beta cutoff
                break

            # Otherwise accumulate
            total_score += score

            # Adjust A, B for next child
            # (From pseudocode: A += U - score, B += L - score)
            A += (U - score)
            B += (L - score)

        if (not pruned_early) and (N > 0):
            total_score /= N

        if depth == origDepth:
            # On a top-level chance node, we can store the first move we didn't prune
            game.ai_move = best_move

        # ---- Store in TT if desired ----
        if tt is not None:
            # Decide if EXACT or LOWERBOUND/UPPERBOUND
            if total_score <= alphaOrig:
                flag = UPPERBOUND
            elif total_score >= beta:
                flag = LOWERBOUND
            else:
                flag = EXACT

            tt.store(game=game, depth=depth, value=total_score, move=best_move, flag=flag)

        return total_score


class ExpectiMinimax:
    """
    A class implementing *-minimax (expectiminimax) with partial alpha-beta,
    mirroring the structure of easyAI's Negamax class.
    """

    def __init__(self, depth, scoring=None, win_score=inf, tt=None):
        self.depth = depth
        self.scoring = scoring
        self.win_score = win_score
        self.tt = tt
        self.alpha = None  # Will store the last computed root-value if desired.

    def __call__(self, game):
        """
        Returns the AI's best move given the current state of the game
        (the move is also stored as game.ai_move inside star_minimax).
        """
        # If no custom scoring was set, use the game's built-in .scoring()
        scoring_fn = self.scoring if self.scoring else (lambda g: g.scoring())

        # We do a negamax-style alpha/beta from -win_score, +win_score
        self.alpha = star_minimax(
            game,
            depth=self.depth,
            origDepth=self.depth,
            scoring=scoring_fn,
            alpha=-self.win_score,
            beta=+self.win_score,
            tt=self.tt
        )
        return game.ai_move
