# rlModelLearner.py - Model-based Reinforcement Learner
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
from rlProblem import RL_agent, Simulate, epsilon_greedy, ucb
from display import Displayable
from utilities import argmaxe, flip

class Model_based_reinforcement_learner(RL_agent):
    """A Model-based reinforcement learner
    """

    def __init__(self, name, actions, discount,
                     exploration_strategy=epsilon_greedy, es_kwargs={},
                     Qinit=0, 
                   updates_per_step=10):
        """name is the name of the agent (e.g., in a game)
        actions is the list of actions the agent can do
        discount is the discount factor
        explore is the proportion of time the agent will explore
        Qinit is the initial value of the Q's
        updates_per_step is the number of AVI updates per action
        label is the label for plotting
        """
        RL_agent.__init__(self, actions)
        self.name = name
        self.actions = actions
        self.discount = discount
        self.exploration_strategy = exploration_strategy
        self.es_kwargs = es_kwargs
        self.Qinit = Qinit
        self.updates_per_step = updates_per_step

    def initial_action(self, state):
        """ Returns the initial action; selected at random
        Initialize Data Structures

        """
        self.action = RL_agent.initial_action(self, state)
        self.T = {self.state: {a: {} for a in self.actions}}
        self.visits = {self.state: {a: 0 for a in self.actions}}
        self.Q = {self.state: {a: self.Qinit for a in self.actions}}
        self.R = {self.state: {a: 0 for a in self.actions}}
        self.states_list = [self.state] # list of states encountered
        self.display(2, f"Initial State: {state} Action {self.action}")
        self.display(2,"s\ta\tr\ts'\tQ")
        return self.action

    def select_action(self, reward, next_state):
        """do num_steps of interaction with the environment
        for each action, do updates_per_step iterations of asynchronous value iteration
        """
        if next_state not in self.visits: # has not been encountered before
            self.states_list.append(next_state)
            self.visits[next_state] = {a:0 for a in self.actions}
            self.T[next_state] = {a:{} for a in self.actions}
            self.Q[next_state] = {a:self.Qinit for a in self.actions}
            self.R[next_state] = {a:0 for a in self.actions}
        if next_state in self.T[self.state][self.action]:
            self.T[self.state][self.action][next_state] += 1
        else:
            self.T[self.state][self.action][next_state] = 1
        self.visits[self.state][self.action] += 1
        self.R[self.state][self.action] += (reward-self.R[self.state][self.action])/self.visits[self.state][self.action]
        st,act = self.state,self.action    #initial state-action pair for AVI
        for update in range(self.updates_per_step):
            self.Q[st][act] = self.R[st][act]+self.discount*(
                sum(self.T[st][act][nst]/self.visits[st][act]*self.v(nst)
                    for nst in self.T[st][act].keys()))
            st = random.choice(self.states_list)
            act = random.choice(self.actions)
        self.state = next_state
        self.action = self.exploration_strategy(next_state, self.Q[next_state],
                                 self.visits[next_state],**self.es_kwargs)
        return self.action

    def q(self, state, action):
        if state in self.Q and action in self.Q[state]:
            return self.Q[state][action]
        else:
            return self.Qinit

from rlExamples import Monster_game_env
mon_env = Monster_game_env()
mbl1 = Model_based_reinforcement_learner("model-based(1)", mon_env.actions, 0.9, updates_per_step=1)
# Simulate(mbl1,mon_env).start().go(100000).plot()
mbl10 = Model_based_reinforcement_learner("model-based(10)", mon_env.actions, 0.9, updates_per_step=10)
# Simulate(mbl10,mon_env).start().go(100000).plot()

from rlGUI import rlGUI
#gui = rlGUI(mon_env, mbl1)

from rlQLearner import test_RL
if __name__ == "__main__":
    test_RL(Model_based_reinforcement_learner)
    
