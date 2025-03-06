import random
import time
import pandas as pd
from easyAI import TwoPlayerGame, AI_Player, Negamax

######################################
# 1) The TicTacToe game class
######################################

class TicTacToe(TwoPlayerGame):
    def __init__(self, players, probabilistic=True):
        """
        :param players: List of two players (AI_Player or Human_Player)
        :param probabilistic: If True, there's a 20% chance that the final move misses.
        """
        self.players = players
        self.current_player = 1
        self.board = [0]*9
        self.probabilistic = probabilistic
        self.simulation = True  # When True => no random skipping during AI search

    def possible_moves(self):
        return [i+1 for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, move):
        # If in simulation or deterministic mode => always place
        if self.simulation or not self.probabilistic:
            self.board[move - 1] = self.current_player
        else:
            # 20% chance to skip
            num = random.random()
            if num < 0.2:
                print(f"Unlucky roll - missed turn! (r={num:.2f})")
            else:
                self.board[move - 1] = self.current_player

    def unmake_move(self, move):
        self.board[move - 1] = 0

    def show(self):
        print()
        for i in range(0, 9, 3):
            row = self.board[i:i+3]
            symbols = [".", "X", "O"]
            print(" ".join(symbols[cell] for cell in row))

    def lose(self):
        """Check if current player lost => the opponent formed 3 in a row."""
        wins = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6]           # diagonals
        ]
        return any(all(self.board[pos] == self.opponent_index for pos in line) for line in wins)

    def is_over(self):
        return self.lose() or not self.possible_moves()

    def scoring(self):
        # -100 if current player is losing, else 0
        return -100 if self.lose() else 0

    def play(self, nmoves=9999, verbose=False):
        """
        Overridden to separate "choose move" (simulation=True) 
        from "apply move" (simulation=False).
        """
        move_history = []
        for _ in range(nmoves):
            if verbose:
                self.show()
            if self.is_over():
                break

            player = self.players[self.current_player - 1]
            # 1) Let the AI/human pick the move (no randomness)
            self.simulation = True
            chosen_move = player.ask_move(self)

            # 2) Apply it "for real" => 20% chance to miss if probabilistic
            self.simulation = False
            self.make_move(chosen_move)
            self.simulation = True

            move_history.append((self.current_player, chosen_move))
            self.switch_player()

        if verbose:
            self.show()
        return move_history


######################################
# 2) Timed AI Player
######################################

class TimedAI_Player(AI_Player):
    """
    Subclass of AI_Player that measures time spent picking moves.
    """
    def __init__(self, ai_algo):
        super().__init__(ai_algo)
        self.total_time = 0.0
        self.n_moves = 0

    def ask_move(self, game):
        start = time.time()
        move = super().ask_move(game)
        end = time.time()

        self.total_time += (end - start)
        self.n_moves += 1
        return move


######################################
# 3) run_matches helper
######################################

def run_matches(n_matches, ai1, ai2, probabilistic=False, verbose=False):
    """
    Runs n_matches TicTacToe games between ai1 and ai2.
    Alternates who starts each match. 
    Returns a dict with:
      - p1_wins, p2_wins, draws
      - p1_avg_time, p2_avg_time
      - p1_starting_wins, p1_non_starting_wins
      - p2_starting_wins, p2_non_starting_wins
    """
    # Overall stats
    p1_wins = 0
    p2_wins = 0
    draws = 0

    # Additional counters for starting vs. non-starting
    p1_starter_wins = 0
    p1_non_starter_wins = 0
    p2_starter_wins = 0
    p2_non_starter_wins = 0

    # Reset timing counters if TimedAI_Player
    for ai in [ai1, ai2]:
        if hasattr(ai, "total_time"):
            ai.total_time = 0.0
            ai.n_moves = 0

    for i in range(n_matches):
        # Switch who starts
        if i % 2 == 0:
            # p1 is players[0], p2 is players[1]
            players = [ai1, ai2]
            physical_p1 = ai1
            physical_p2 = ai2
            # So p1 is the "starter" this game
            p1_starts = True
        else:
            # p2 is players[0], p1 is players[1]
            players = [ai2, ai1]
            physical_p1 = ai1
            physical_p2 = ai2
            # p2 is the "starter" this game
            p1_starts = False

        game = TicTacToe(players, probabilistic=probabilistic)
        game.play(verbose=verbose)

        # If .lose() is True, current_player lost => game.opponent_index is the winner
        if game.lose():
            winner = game.opponent_index  # 1 or 2
            # winner == 1 => players[0], winner == 2 => players[1]
            if winner == 1:
                # physically, winner is players[0]
                if players[0] == physical_p1:
                    # p1 is the winner
                    p1_wins += 1
                    if p1_starts:
                        p1_starter_wins += 1
                    else:
                        p1_non_starter_wins += 1
                else:
                    # p2 is the winner
                    p2_wins += 1
                    if not p1_starts:
                        # Then p2 started
                        p2_starter_wins += 1
                    else:
                        p2_non_starter_wins += 1
            else:
                # winner == 2 => players[1]
                if players[1] == physical_p1:
                    # p1 is the winner
                    p1_wins += 1
                    if p1_starts:
                        p1_starter_wins += 1
                    else:
                        p1_non_starter_wins += 1
                else:
                    # p2 is the winner
                    p2_wins += 1
                    if not p1_starts:
                        p2_starter_wins += 1
                    else:
                        p2_non_starter_wins += 1
        else:
            draws += 1

    # Build results
    results = {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_starting_wins": p1_starter_wins,
        "p1_non_starting_wins": p1_non_starter_wins,
        "p2_starting_wins": p2_starter_wins,
        "p2_non_starting_wins": p2_non_starter_wins,
    }

    # If TimedAI, also record average times
    for idx, ai in enumerate([ai1, ai2], start=1):
        if hasattr(ai, "total_time") and ai.n_moves > 0:
            avg_time = ai.total_time / ai.n_moves
            results[f"p{idx}_avg_time"] = avg_time
        else:
            results[f"p{idx}_avg_time"] = None

    return results


######################################
# 4) Main function
######################################

def main():
    # We'll test these pairs of depths
    depth_pairs = [
        (3, 3),
        (9, 9),
        (3, 9),
    ]

    # For alpha-beta, we use win_score=float('inf')
    # For "no alpha-beta," we pass a very large finite bound => effectively no pruning
    def build_timed_negamax(depth, use_ab=True):
        if use_ab:
            return TimedAI_Player(Negamax(depth=depth, scoring=None, win_score=float('inf'), tt=None))
        else:
            return TimedAI_Player(Negamax(depth=depth, scoring=None, win_score=1e9, tt=None))

    results_list = []

    def record_results(label_ab, depth1, depth2, prob, res_dict):
        row = {
            "AlphaBeta": label_ab,  # "AB" or "NoAB"
            "Depth1": depth1,
            "Depth2": depth2,
            "Probabilistic": prob,
            "P1_Wins": res_dict["p1_wins"],
            "P2_Wins": res_dict["p2_wins"],
            "Draws": res_dict["draws"],
            "P1_Starting_Wins": res_dict["p1_starting_wins"],
            "P1_NonStarting_Wins": res_dict["p1_non_starting_wins"],
            "P2_Starting_Wins": res_dict["p2_starting_wins"],
            "P2_NonStarting_Wins": res_dict["p2_non_starting_wins"],
            "P1_AvgTime": res_dict["p1_avg_time"],
            "P2_AvgTime": res_dict["p2_avg_time"],
        }
        results_list.append(row)

    n_matches = 100

    for (d1, d2) in depth_pairs:
        for ab_setting in [True, False]:
            # Create two AIs
            ai1 = build_timed_negamax(d1, use_ab=ab_setting)
            ai2 = build_timed_negamax(d2, use_ab=ab_setting)
            label = "AB" if ab_setting else "NoAB"

            # 1) Deterministic
            res_det = run_matches(n_matches, ai1, ai2, probabilistic=False, verbose=False)
            record_results(label, d1, d2, False, res_det)

            # 2) Probabilistic
            # Rebuild so the times reset
            ai1 = build_timed_negamax(d1, use_ab=ab_setting)
            ai2 = build_timed_negamax(d2, use_ab=ab_setting)
            res_prob = run_matches(n_matches, ai1, ai2, probabilistic=True, verbose=False)
            record_results(label, d1, d2, True, res_prob)

    df = pd.DataFrame(results_list)
    print(df)

    df.to_excel("results.xlsx", index=False)
    print("Results saved to results.xlsx")


if __name__ == "__main__":
    main()
