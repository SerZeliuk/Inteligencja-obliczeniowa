# rlFeatures.py - Feature-based Reinforcement Learner
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
from rlProblem import RL_agent, epsilon_greedy, ucb
from display import Displayable
from utilities import argmaxe, flip
import rlGameFeature

class SARSA_LFA_learner(RL_agent):
    """A SARSA with linear function approximation (LFA) learning agent has
    """
    def __init__(self, name, actions, discount, get_features=rlGameFeature.party_features4,
                     exploration_strategy=epsilon_greedy, es_kwargs={},
                     step_size=0.01, winit=0):
        """name is the name of the agent (e.g., in a game)
        actions is the set of actions the agent can do
        discount is the discount factor
        get_features is a function get_features(state,action) -> list of feature values
        exploration_strategy is the exploration function, default "epsilon_greedy"
        es_kwargs is extra keyword arguments of the exploration_strategy 
        step_size is gradient descent step size
        winit is the initial value of the weights
        """
        RL_agent.__init__(self, actions)
        self.name = name
        self.discount = discount
        self.exploration_strategy = exploration_strategy
        self.es_kwargs = es_kwargs
        self.get_features = get_features
        self.step_size = step_size
        self.winit = winit

    def initial_action(self, state):
        """ Returns the initial action; selected at random
        Initialize Data Structures
        """
        self.action = RL_agent.initial_action(self, state)
        self.features = self.get_features(state, self.action)
        self.weights = [self.winit for f in self.features]
        self.display(2, f"Initial State: {state} Action {self.action}")
        self.display(2,"s\ta\tr\ts'\tQ")
        return self.action


    def q(self, state,action):
        """returns Q-value of the state and action for current weights
        """
        return dot_product(self.weights, self.get_features(state,action))
        
    def select_action(self, reward, next_state):
        """do num_steps of interaction with the environment"""
        feature_values = self.get_features(self.state,self.action)
        oldQ = self.q(self.state,self.action)
        next_action = self.exploration_strategy(next_state, {a:self.q(next_state,a)
                                                     for a in self.actions}, {})
        nextQ = self.q(next_state,next_action)
        delta = reward + self.discount * nextQ - oldQ
        for i in range(len(self.weights)):
            self.weights[i] += self.step_size * delta * feature_values[i]
        self.display(2,self.state, self.action, reward, next_state,
                     self.q(self.state,self.action), delta, sep='\t')
        self.state = next_state
        self.action = next_action
        return self.action

    def show_actions(self,state=None):
        """prints the value for each action in a state.
        This may be useful for debugging.
        """
        if state is None:
            state = self.state
        for next_act in self.actions:
            print(next_act,dot_product(self.weights, self.get_features(state,next_act)))

def dot_product(l1,l2):
    return sum(e1*e2 for (e1,e2) in zip(l1,l2))

from rlProblem import Simulate
from rlExamples import Party_env, Monster_game_env
import rlGameFeature
from rlGUI import rlGUI 

party = Party_env()
pa3 = SARSA_LFA_learner(party.name, party.actions, 0.9, rlGameFeature.party_features3)
# Simulate(pa3,party).start().go(300).plot()
pa4 = SARSA_LFA_learner(party.name, party.actions, 0.9, rlGameFeature.party_features4)
# Simulate(pa4,party).start().go(300).plot()

mon_env = Monster_game_env()
fa1 = SARSA_LFA_learner("LFA", mon_env.actions, 0.9, rlGameFeature.monster_features)
# Simulate(fa1,mon_env).start().go(100000).plot()
fas1 = SARSA_LFA_learner("LFA (simp features)", mon_env.actions, 0.9, rlGameFeature.simp_features)
#Simulate(fas1,mon_env).start().go(100000).plot()
# rlGUI(mon_env, SARSA_LFA_learner(mon_env.name, mon_env.actions, 0.9, rlGameFeature.monster_features))

from rlQLearner import test_RL
if __name__ == "__main__":
    test_RL(SARSA_LFA_learner, es_kwargs={'epsilon':1}) # random exploration
    
