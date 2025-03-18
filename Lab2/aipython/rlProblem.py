# rlProblem.py - Representations for Reinforcement Learning
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
import math
from display import Displayable
from agents import Agent, Environment
from utilities import select_from_dist, argmaxe, argmaxd, flip

class RL_env(Environment):
    def __init__(self, name, actions, state):
        """creates an environment given name, list of actions, and initial state"""
        self.name = name         # the name of the environment
        self.actions = actions   # list of all actions
        self.state = state       # initial state
        self.reward = None       # last reward

    # must implement do(action)->(reward,state)
    
class RL_agent(Agent):
    """An RL_Agent 
    has percepts (s, r) for some state s and real reward r
    """
    def __init__(self, actions):
       self.actions = actions

    def initial_action(self, env_state):
        """return the initial action, and remember the state and action
        Act randomly initially
        Could be overridden to initialize data structures (as the agent now knows about one state)
        """
        self.state = env_state
        self.action = random.choice(self.actions)
        return self.action

    def select_action(self, reward, state):
        """ 
        Select the action given the reward and state
        Remember the action in self.action
        This implements "Act randomly" and should  be overridden!
        """
        self.reward = reward
        self.action = random.choice(self.actions)
        return self.action

    def v(self, state):
        """estimate of the value of doing a best action in state.
        """
        return max(self.q(state,a) for a in self.actions)

    def q(self, state, action):
        """"estimate of value of doing action in state. Should be overridden to be useful.
        """
        return 0

import matplotlib.pyplot as plt

class Simulate(Displayable):
    """simulate the interaction between the agent and the environment
    for n time steps.
    Returns a pair of the agent state and the environment state.
    """
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.reward_history = []  # for plotting
        self.step = 0
        self.sum_rewards = 0

    def start(self):
        self.action = self.agent.initial_action(self.env.state)
        return self

    def go(self, n):
        for i in range(n):
            self.step += 1
            (reward,state) = self.env.do(self.action)
            self.display(2,f"step={self.step} reward={reward}, state={state}")
            self.sum_rewards += reward
            self.reward_history.append(reward)
            self.action = self.agent.select_action(reward,state)
            self.display(2,f"      action={self.action}")
        return self
    def plot(self, label=None, step_size=None, xscale='linear'):
        """
        plots the rewards history in the simulation
        label is the label for the plot
        step_size is the number of steps between each point plotted
        xscale is 'log' or 'linear'

        returns sum of rewards
        """
        if step_size is None: #for long simulations (> 999), only plot some points
            step_size = max(1,len(self.reward_history)//500)
        if label is None:
            label = self.agent.name
        plt.ion()
        plt.xscale(xscale)
        plt.title(self.env.name)
        plt.xlabel("step")
        plt.ylabel("Sum of rewards")
        sum_history, sum_rewards = acc_rews(self.reward_history, step_size)
        plt.plot(range(0,len(self.reward_history),step_size), sum_history, label=label)
        plt.legend()
        plt.draw()
        return sum_rewards

def acc_rews(rews,step_size):
    """returns the rolling sum of the values, sampled each step_size, and the sum
    """
    acc = []
    sumr = 0; i=0
    for e in rews:
       sumr += e
       i += 1
       if (i%step_size == 0): acc.append(sumr)
    return acc, sumr


class Env_from_ProblemDomain(RL_env):
    def __init__(self, prob_dom):
        RL_env.__init__(self, prob_dom.title, prob_dom.actions, prob_dom.state)
        self.problem_domain = prob_dom
        self.state = prob_dom.state
        self.x_dim = prob_dom.x_dim
        self.y_dim = prob_dom.y_dim 
        self.offsets = prob_dom.offsets
        self.state2pos = self.problem_domain.state2pos
        self.state2goal = self.problem_domain.state2goal
        self.pos2state = self.problem_domain.pos2state
        
    def do(self, action):
        """updates the state based on the agent doing action.
        returns state,reward
        """
        (self.reward, self.state) = select_from_dist(self.problem_domain.result(self.state, action))
        self.problem_domain.state = self.state
        self.display(2,f"do({action} -> ({self.reward}, {self.state})")
        return (self.reward,self.state)

def epsilon_greedy(state, Qs, visits={}, epsilon=0.2):
        """select action given epsilon greedy
        Qs is the {action:Q-value} dictionary for current state
        visits is ignored
        epsilon is the probability of acting randomly
        """
        if flip(epsilon):
            return random.choice(list(Qs.keys())) # act randomly
        else:
            return argmaxd(Qs) # pick an action with max Q

def ucb(state, Qs, visits, c=1.4):
        """select action given upper-confidence bound
        Qs is the  {action:Q-value} dictionary for current state
        visits is the {action:n} dictionary for current state

        0.01 is to prevent divide-by zero when visits[a]==0
        """
        Ns = sum(visits.values())
        ucb1 = {a:Qs[a]+c*math.sqrt(Ns/(0.01+visits[a]))
                    for a in Qs.keys()}
        action = argmaxd(ucb1)
        return action

