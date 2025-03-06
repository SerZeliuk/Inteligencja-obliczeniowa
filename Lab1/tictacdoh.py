from math import ceil, floor
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import random

class TicTacToe(TwoPlayerGame):
    def __init__(self, players, prbl=True):
        self.players = players
        self.board = [0 for i in range(9)]  # Initialize empty board
        self.current_player = 1  # Player 1 starts
        self.probabilistic = prbl

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        # print("its a move MADE by ", self.players[self.current_player - 1].name) 
        # if self.probabilistic:    
        #     rand_num = int(round(random.random(), 2)*100)
            
        #     if ceil(rand_num) > 20:
                
        #         self.board[int(move) - 1] = self.current_player
        #     else:
        #         if self.players[self.current_player - 1].name == "AI":
        #             print(f"Unlucky roll, you silly {self.players[self.current_player - 1].name} - you get {rand_num}, you miss your turn.")
        #         else:
        #             pass
        # else:
        self.board[int(move) - 1] = self.current_player

            
    def unmake_move(self, move):
        print("Unmaking a move")
        self.board[int(move) - 1] = 0

    def show(self):
        print("\n")
        for i in range(3):
            for j in range(3):
                if self.board[3*i + j] == 0:
                    print(".", end=" ")
                elif self.board[3*i + j] == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()

    def lose(self):
        return any(all(self.board[c-1] == self.opponent_index 
                      for c in line)
                  for line in [[1,2,3], [4,5,6], [7,8,9],  # horizontal
                             [1,4,7], [2,5,8], [3,6,9],    # vertical
                             [1,5,9], [3,5,7]])            # diagonal

    def is_over(self):
        return (self.lose() or 
                len(self.possible_moves()) == 0)

    def scoring(self):
        return -100 if self.lose() else 0

if __name__ == "__main__":
    
    ai_algo = Negamax(9) 
    game = TicTacToe([AI_Player(ai_algo), Human_Player()])
    
    
    # while not game.is_over():
    #     game.show()
    #     print(f"\nPlayer {game.current_player}'s turn")
    #     if game.current_player == 1:
    #         print("AI is thinking...")
    #     else:
    #         print(f"Possible moves: {game.possible_moves()}")
    #     game.play()

    # # Show final state
    # game.show()
    history = game.play()

    if game.lose():
        print(f"\nPlayer {game.opponent_index} wins!")
    else:
        print("\nIt's a draw!")