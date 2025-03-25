# masLearn.py - Multiagent learning
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
from display import Displayable
import matplotlib.pyplot as plt
from rlProblem import RL_agent

class Game(Displayable):
    def __init__(self, name, players, actions, states=['s0'], initial_state=None):
        self.name = name
        self.players = players  # list of roles (strings) of the players
        self.num_players = len(players)
        self.actions = actions # action[i] is list of actions for agent i
        self.states = states   # list of environment states; default single state
        if initial_state is None:
            self.initial_state = random.choice(states)
        else:
            self.initial_state = initial_state

    def sim(self, ag_types, discount=0):
        """returns a simulation using default values for agent types
             (This is a simple interface to SimulateGame)
          ag_types is a list of agent functions (one for each player in the game)
             The default is for one-off games where discount=0
        """
        return SimulateGame(self,
                            [ag_types[i](ag_types[i].__name__, self.actions[i], discount)
                               for i in range(self.num_players)])

class SimulateGame(Displayable):
    """A simulation of a game. 
       (This is not subclass of a game, as a game can have multiple games.)
    """
    def __init__(self, game, agents):
        """ Simulates game
            agents is a list of agents, one for each player in the game
        """
        #self.max_display_level = 3
        self.game = game
        self.agents = agents
        # Collect Statistics:
        self.action_counts = [{act:0 for act in game.actions[i]} for i in range(game.num_players)]
        self.reward_sum = [0 for i in range(game.num_players)]
        self.dist = {}
        self.dist_history = []
        self.actions = tuple(ag.initial_action(game.initial_state) for ag in self.agents)
        self.num_steps = 0

    def go(self, steps):
        for i in range(steps):
            self.num_steps += 1
            (rewards, state) = self.game.play(self.actions)
            self.display(3, f"In go {rewards=}, {state=}")
            self.reward_sum = [self.reward_sum[i]+rewards[i] for i in range(len(rewards))]
            self.actions = tuple(agent.select_action(reward, state)
                                     for (agent,reward) in zip(self.agents,rewards))
            for i in range(self.game.num_players):
                 self.action_counts[i][self.actions[i]] += 1
            self.dist_history.append([{a:i/self.num_steps for (a,i) in elt.items()}
                                          for elt in self.action_counts])
        self.display(1,"Scores:", ' '.join(
            f"{self.agents[i].name} average reward={self.reward_sum[i]/self.num_steps}"
                                for i in range(self.game.num_players)))
        self.display(1,"Distributions:",
                         ' '.join(str({a:self.dist_history[-1][i][a]
                                           /sum(self.dist_history[-1][i].values())
                                        for a in self.game.actions[i]})
                                    for  i in range(self.game.num_players)))

    def plot_dynamics(self, x_ag=0, y_ag=1, x_action=0, y_action=0):
        """ plot how the empirical probabilities vary 
        x_ag index of the agent on the x-axis
        y_ag index of the agent on the y-axis
        x_action index of the action plotted for x_ag
        y_action index of the action plotted for y_ag
        """
        plt.ion()  # make it interactive
        plt.title(self.game.name)
        x_act = self.game.actions[x_ag][x_action]
        y_act = self.game.actions[y_ag][y_action]
        plt.xlabel(f"Probability {self.game.players[x_ag]} does "
                       f"{self.agents[x_ag].actions[x_action]}")
        plt.ylabel(f"Probability {self.game.players[y_ag]} does "
                       f"{self.agents[y_ag].actions[y_action]}")
        plt.plot([self.dist_history[i][x_ag][x_act]
                    for i in range(len(self.dist_history))],
                 [self.dist_history[i][y_ag][y_act]
                    for i in range(len(self.dist_history))],
                 label = f"({self.agents[x_ag].name}, {self.agents[y_ag].name})")
        plt.legend()
        plt.show()

class ShoppingGame(Game):
    def __init__(self):
        Game.__init__(self, "Shopping Game",
                      ['football-preferrer', 'shopping-preferrer'], #players
                      [['shopping', 'football']]*2  # actions
                      )

    def play(self, actions):
        """Given (action1,action2) returns (resulting_state, (reward1, reward2))
        """
        return ({('football', 'football'): (2, 1),
                 ('football', 'shopping'): (0, 0),
                 ('shopping', 'football'): (0, 0),
                 ('shopping', 'shopping'): (1, 2)
                     }[actions], 's')

class SoccerGame(Game):
    def __init__(self):
        Game.__init__(self, "Soccer Gaol Kick Game",
                          ['goalkeeper', 'kicker'], # players
                          [['right', 'left']]*2   # actions
                      )
        
    def play(self, actions):
        """Given (action1,action2) returns (resulting_state, (reward1, reward2))
        resulting state is 's'
        """
        return ({('left', 'left'): (0.6, 0.4),
                 ('left', 'right'): (0.3, 0.7),
                 ('right', 'left'): (0.2, 0.8),
                 ('right', 'right'): (0.9,0.1)
               }[actions], 's')
               
class GameShow(Game):
    def __init__(self):
        Game.__init__(self, "Game Show (prisoners dilemma)",
                          ['Agent 1', 'Agent 2'], # players
                          [['takes', 'gives']]*2   # actions
                      )

    def play(self, actions):
        return ({('takes', 'takes'): (1, 1),
                ('takes', 'gives'): (11, 0),
                ('gives', 'takes'): (0, 11),
                ('gives', 'gives'): (10, 10)
               }[actions], 's')
               
class UniqueNEGameExample(Game):
    def __init__(self):
        Game.__init__(self, "3x3 Unique NE Game Example",
                      ['agent 1', 'agent 2'], # players    
                      [['a1', 'b1', 'c1'],['d2', 'e2', 'f2']]
                     )
        
    def play(self, actions):
        return ({('a1', 'd2'): (3, 5),
                 ('a1', 'e2'): (5, 1),
                 ('a1', 'f2'): (1, 2),
                 ('b1', 'd2'): (1, 1),
                 ('b1', 'e2'): (2, 9),
                 ('b1', 'f2'): (6, 4),
                 ('c1', 'd2'): (2, 6),
                 ('c1', 'e2'): (4, 7),
                 ('c1', 'f2'): (0, 8)
                     }[actions], 's')

# Choose a game:
# gm = ShoppingGame()
# gm = SoccerGame()
# gm = GameShow()
# gm = UniqueNEGameExample()

from rlQLearner import Q_learner
from rlProblem import RL_agent
from rlStochasticPolicy import StochasticPIAgent
# Choose one of the combinations of learners:
# sm = gm.sim([StochasticPIAgent, StochasticPIAgent]); sm.go(10000)
# sm = gm.sim([Q_learner, Q_learner]); sm.go(10000)
# sm = gm.sim([Q_learner, StochasticPIAgent]); sm.go(10000)
# sm = gm.sim([StochasticPIAgent, Q_learner]); sm.go(10000)

# sm.plot_dynamics()

