# rlStochasticPolicy.py - Simulations of agents learning
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
import utilities  # argmaxall for (element,value) pairs
import matplotlib.pyplot as plt
import random
from rlQLearner import Q_learner

class StochasticPIAgent(Q_learner):
    """This agent  maintains the Q-function for each state. 
    Chooses the best action using empirical distribution over actions
    """
    def __init__(self, name, actions, discount=0, pi_init=1, **nargs):
        """
        name is the name of the agent (e.g., in a game)
        actions is the set of actions the agent can do.
        discount is the discount factor (0 is appropriate if there is a single state)
        pi_init gives the prior counts (Dirichlet prior) for the policy (must be >0)
        """
        #self.max_display_level = 3
        Q_learner.__init__(self, name, actions, discount,
                            exploration_strategy=self.action_from_stochastic_policy,
                            **nargs)
        self.pi_init = pi_init
        self.pi = {}

    def initial_action(self, state):
        """ update policy pi then do initial action from Q_learner
        """
        self.pi[state] = {act:self.pi_init for act in self.actions}
        return Q_learner.initial_action(self, state)
    
    def action_from_stochastic_policy(self, next_state, qs, vs):
         a_best = utilities.argmaxd(self.Q[self.state])
         self.pi[self.state][a_best] +=1
         if next_state not in self.pi:
             self.pi[next_state] = {act:self.pi_init for act in self.actions}
         return select_from_dist(self.pi[next_state])
         
def normalize(dist):
    """dict is a {value:number} dictionary, where the numbers are all non-negative
    returns dict where the numbers sum to one
    """
    tot = sum(dist.values())
    return {var:val/tot for (var,val) in dist.items()}

def select_from_dist(dist):
    rand = random.random()
    for (act,prob) in normalize(dist).items():
        rand -= prob
        if rand < 0:
            return act

#### Testing on RL benchmarks #####
from rlProblem import Simulate
import rlExamples
mon_env = rlExamples.Monster_game_env()
magspi =StochasticPIAgent(mon_env.name, mon_env.actions,0.9)
#Simulate(magspi,mon_env).start().go(100000).plot()
magspi10 = StochasticPIAgent("stoch 10/(9+k)", mon_env.actions,0.9, alpha_fun=lambda k:10/(9+k))
#Simulate(magspi10,mon_env).start().go(100000).plot()

from rlQLearner import test_RL
if __name__ == "__main__":
    test_RL(StochasticPIAgent, alpha_fun=lambda k:10/(9+k))
    
