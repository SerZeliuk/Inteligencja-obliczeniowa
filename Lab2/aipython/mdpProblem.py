# mdpProblem.py - Representations for Markov Decision Processes
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
from display import Displayable
from utilities import argmaxd

class MDP(Displayable):
    """A Markov Decision Process. Must define:
    title a string that gives the title of the MDP
    states the set (or list) of states
    actions the set (or list) of actions
    discount a real-valued discount
    """

    def __init__(self, title, states, actions, discount, init=0):
        self.title = title
        self.states = states
        self.actions = actions
        self.discount = discount
        self.initv = self.V = {s:init for s in self.states}
        self.initq = self.Q = {s: {a: init for a in self.actions} for s in self.states}

    def P(self,s,a):
        """Transition probability function
        returns a dictionary of {s1:p1} such that P(s1 | s,a)=p1,
                 and other probabilities are zero.
        """
        raise NotImplementedError("P")   # abstract method

    def R(self,s,a):
        """Reward function R(s,a)
        returns the expected reward for doing a in state s.
        """
        raise NotImplementedError("R")   # abstract method

class distribution(dict):
    """A distribution is an item:prob dictionary.
    Probabilities are added using add_prop. 
    """
    def __init__(self,d):
        dict.__init__(self,d)

    def add_prob(self, item, pr):
        """adds a probability to a distribution.
        Like dictionary assignment, but if item is already there, the values are summed
        """
        if item in self:
            self[item] += pr
        else:
            self[item] = pr
        return self 
            
class ProblemDomain(MDP):
    """A ProblemDomain implements
    self.result(state, action) -> {(reward, state):probability}. 
    Other pairs have probability are zero.
    The probabilities must sum to 1.
    """
    def __init__(self, title, states, actions, discount,
                     initial_state=None, x_dim=0, y_dim = 0,
                     vinit=0, offsets={}):
        """A problem domain
        * title is list of titles
        * states is the list of states
        * actions is the list of actions
        * discount is the discount factor
        * initial_state is the state the agent starts at (for simulation) if known
        * x_dim and y_dim are the dimensions used by the GUI to show the states in 2-dimensions
        * vinit is the initial value
        * offsets is a {action:(x,y)} map which specifies how actions are displayed in GUI
        """
        MDP.__init__(self, title, states, actions, discount)
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = random.choice(states)
        self.vinit = vinit # value to reset v,q to
        # The following are for the GUI:
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.offsets = offsets

    def state2pos(self,state):
        """When displaying as a grid, this specifies how the state is mapped to (x,y) position.
        The default is for domains where the (x,y) position is the state
        """
        return state
        
    def state2goal(self,state):
        """When displaying as a grid, this specifies how the state is mapped to goal position.
        The default is for domains where there is no goal
        """
        return None
        
    def pos2state(self,pos):
        """When displaying as a grid, this specifies how the state is mapped to (x,y) position.
        The default is for domains where the (x,y) position is the state
        """
        return pos

    def P(self, state, action):
        """Transition probability function
        returns a dictionary of {s1:p1} such that P(s1 | state,action)=p1. 
        Other probabilities are zero.
        """
        res = self.result(state, action)
        acc = 1e-6  # accuracy for test of equality
        assert 1-acc<sum(res.values())<1+acc, f"result({state},{action}) not a distribution, sum={sum(res.values())}"
        dist = distribution({}) 
        for ((r,s),p) in res.items():
            dist.add_prob(s,p)
        return dist

    def R(self, state, action):
        """Reward function R(s,a)
        returns the expected reward for doing a in state s.
        """
        return sum(r*p for ((r,s),p) in self.result(state, action).items())

def vi(self,  n):
        """carries out n iterations of value iteration, updating value function self.V
        Returns a Q-function, value function, policy
        """
        self.display(3,f"calling vi({n})")
        for i in range(n):
            self.Q = {s: {a: self.R(s,a)
                            +self.discount*sum(p1*self.V[s1]
                                                for (s1,p1) in self.P(s,a).items())
                          for a in self.actions}
                     for s in self.states}
            self.V = {s: max(self.Q[s][a] for a in self.actions)
                      for s in self.states}
        self.pi = {s: argmaxd(self.Q[s])
                      for s in self.states}
        return self.Q, self.V, self.pi

MDP.vi = vi

def avi(self,n):
          states = list(self.states)
          actions = list(self.actions)
          for i in range(n):
              s = random.choice(states)
              a = random.choice(actions)
              self.Q[s][a] = (self.R(s,a) + self.discount *
                                  sum(p1 * max(self.Q[s1][a1]
                                                    for a1 in self.actions)
                                        for (s1,p1) in self.P(s,a).items()))
          return self.Q

# make this a method for the MPD class:
MDP.avi = avi

