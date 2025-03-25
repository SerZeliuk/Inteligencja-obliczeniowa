# rlQExperienceReplay.py - Q-Learner with Experience Replay
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from rlQLearner import Q_learner
from utilities import flip
import random

class BoundedBuffer(object):
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = [0]*buffer_size
        self.number_added = 0

    def add(self, new_value):
        if self.number_added < self.buffer_size:
            self.buffer[self.number_added] = new_value
        else:
            if flip(self.buffer_size/self.number_added):
                position = random.randrange(self.buffer_size)
                self.buffer[position] = new_value
        self.number_added += 1

    def get(self):
        return self.buffer[random.randrange(min(self.number_added, self.buffer_size))]

class Q_ER_learner(Q_learner):
    def __init__(self, name, actions, discount, 
                 max_buffer_size=10000,
                 num_updates_per_action=10, burn_in=100, **q_kwargs):
        """Q-learner with experience replay
        name is the name of the agent (e.g., in a game)
        actions is the set of actions the agent can do
        discount is the discount factor
        max_buffer_size is the maximum number of past experiences that is remembered
        burn_in is the number of steps before using old experiences
        num_updates_per_action is the number of q-updates for past experiences per action
        q_kwargs are any extra parameters for Q_learner
        """
        Q_learner.__init__(self, name, actions, discount, **q_kwargs)
        self.experience_buffer = BoundedBuffer(max_buffer_size)
        self.num_updates_per_action = num_updates_per_action
        self.burn_in = burn_in

    def select_action(self, reward, next_state):
        """give reward and new state, select next action to be carried out"""
        self.experience_buffer.add((self.state,self.action,reward,next_state)) #remember experience
        if next_state not in self.visits:  # next_state not seen before
            self.Q[next_state] = {act:self.Qinit for act in self.actions}
            self.visits[next_state] = {act:0 for act in self.actions}
        self.visits[self.state][self.action] +=1
        alpha = self.alpha_fun(self.visits[self.state][self.action])
        self.Q[self.state][self.action] += alpha*(
                            reward
                            + self.discount * max(self.Q[next_state].values())
                            - self.Q[self.state][self.action])
        self.display(2,self.state, self.action, reward, next_state, 
                     self.Q[self.state][self.action], sep='\t')
        # do some updates from experience buffer
        if self.experience_buffer.number_added > self.burn_in:
            for i in range(self.num_updates_per_action):
                (s,a,r,ns) = self.experience_buffer.get()
                self.visits[s][a] +=1   # is this correct?
                alpha = self.alpha_fun(self.visits[s][a])
                self.Q[s][a] += alpha * (r + 
                                    self.discount* max(self.Q[ns][na]
                                            for na in self.actions)
                                    -self.Q[s][a] )
        ### CHOOSE NEXT ACTION ###
        self.action = self.exploration_strategy(next_state, self.Q[next_state],
                                        self.visits[next_state],**self.es_kwargs)
        self.state = next_state
        self.display(3,f"Agent {self.name} doing {self.action} in state {self.state}")
        return self.action

from rlProblem import Simulate
from rlExamples import Monster_game_env
from rlQLearner import mag1, mag2, mag3

mon_env = Monster_game_env()
mag1ar = Q_ER_learner("Q_ER", mon_env.actions,0.9,
                          num_updates_per_action=5, burn_in=100)
# Simulate(mag1ar,mon_env).start().go(100000).plot() 

mag3ar = Q_ER_learner("Q_ER alpha=10/(9+k)", mon_env.actions, 0.9,
                          num_updates_per_action=50, burn_in=1000,
                          alpha_fun=lambda k:10/(9+k))
# Simulate(mag3ar,mon_env).start().go(100000).plot()

from rlQLearner import test_RL
if __name__ == "__main__":
    test_RL(Q_ER_learner, alpha_fun=lambda k:10/(9+k))
    
