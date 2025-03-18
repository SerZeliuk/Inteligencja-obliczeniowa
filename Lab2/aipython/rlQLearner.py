# rlQLearner.py - Q Learning
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
from utilities import argmaxe, argmaxd, flip
from rlProblem import RL_agent, epsilon_greedy, ucb

class Q_learner(RL_agent):
    """A Q-learning agent has
    belief-state consisting of
        state is the previous state (initialized by RL_agent
        q is a {(state,action):value} dict
        visits is a {(state,action):n} dict.  n is how many times action was done in state
        acc_rewards is the accumulated reward
    """
    
    def __init__(self, name, actions, discount,
                 exploration_strategy=epsilon_greedy, es_kwargs={},
                 alpha_fun=lambda _:0.2, Qinit=0):
        """
        name is string representation of the agent
        actions is the set of actions the agent can do
        discount is the discount factor
        exploration_strategy is the exploration function, default "epsilon_greedy"
        es_kwargs is extra arguments of exploration_strategy 
        alpha_fun is a function that computes alpha from the number of visits
        Qinit is the initial q-value
        """
        RL_agent.__init__(self, actions)
        self.name = name
        self.discount = discount
        self.exploration_strategy = exploration_strategy
        self.es_kwargs = es_kwargs
        self.alpha_fun = alpha_fun
        self.Qinit = Qinit
        self.acc_rewards = 0
        self.Q = {}
        self.visits = {}

    def initial_action(self, state):
        """ Returns the initial action; selected at random
        Initialize Data Structures
        """
        self.state = state
        self.Q[state] = {act:self.Qinit for act in self.actions}
        self.visits[state] = {act:0 for act in self.actions}
        self.action = self.exploration_strategy(state, self.Q[state],
                                     self.visits[state],**self.es_kwargs)
        self.display(2, f"Initial State: {state} Action {self.action}")
        self.display(2,"s\ta\tr\ts'\tQ")
              # display looks best if states and actions are < 8 characters
        return self.action
        
    def select_action(self, reward, next_state):
        """give reward and next state, select next action to be carried out"""
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
        self.action = self.exploration_strategy(next_state, self.Q[next_state],
                                     self.visits[next_state],**self.es_kwargs)
        self.state = next_state
        self.display(3,f"Agent {self.name} doing {self.action} in state {self.state}")
        return self.action

    def q(self,s,a):
        if s in self.Q and a in self.Q[s]:
            return self.Q[s][a]
        else:
            return self.Qinit
            
class SARSA(Q_learner):
    def __init__(self,*args, **nargs):
        Q_learner.__init__(self,*args, **nargs)
        
    def select_action(self, reward, next_state):
        """give reward and next state, select next action to be carried out"""
        if next_state not in self.visits:  # next state not seen before
            self.Q[next_state] = {act:self.Qinit for act in self.actions}
            self.visits[next_state] = {act:0 for act in self.actions}
        self.visits[self.state][self.action] +=1
        alpha = self.alpha_fun(self.visits[self.state][self.action])
        next_action = self.exploration_strategy(next_state, self.Q[next_state],
                                     self.visits[next_state],**self.es_kwargs)
        self.Q[self.state][self.action] += alpha*(
                            reward
                            + self.discount * self.Q[next_state][next_action]
                            - self.Q[self.state][self.action])
        self.display(2,self.state, self.action, reward, next_state, 
                     self.Q[self.state][self.action], sep='\t')
        self.state = next_state
        self.action = next_action
        self.display(3,f"Agent {self.name} doing {self.action} in state {self.state}")
        return self.action

####### TEST CASES ########
from rlProblem import Simulate,epsilon_greedy, ucb, Env_from_ProblemDomain
from rlExamples import Party_env, Monster_game_env
from rlQLearner import Q_learner
from mdpExamples import MDPtiny, partyMDP

def test_RL(learnerClass, mdp=partyMDP, env=Party_env(), discount=0.9, eps=2, **lkwargs):
    """tests whether RL on env has the same (within eps) Q-values as vi on mdp"""
    mdp1 = mdp(discount=discount)
    q1,v1,pi1 = mdp1.vi(1000)
    ag = learnerClass(learnerClass.__name__, env.actions, discount, **lkwargs)
    sim = Simulate(ag,env).start()
    sim.go(100000)
    same = all(abs(ag.q(s,a)-q1[s][a]) < eps
                   for s in mdp1.states
                   for a in mdp1.actions)
    assert same, (f"""Unit test failed for {env.name}, in {ag.name} Q="""
                    +str({(s,a):ag.q(s,a) for s in mdp1.states
                                    for a in mdp1.actions})
                    +f""" in vi Q={q1}""")
    print(f"Unit test passed. For {env.name}, {ag.name} has same Q-value as value iteration")
if __name__ == "__main__":
    test_RL(Q_learner, alpha_fun=lambda k:10/(9+k))
    #test_RL(SARSA) # should this pass? Why or why not?

#env = Party_env()
env = Env_from_ProblemDomain(MDPtiny())
# Some RL agents with different parameters:
ag = Q_learner("eps (0.1) greedy", env.actions, 0.7)
ag_ucb = Q_learner("ucb", env.actions, 0.7, exploration_strategy = ucb, es_kwargs={'c':0.1})
ag_opt = Q_learner("optimistic", env.actions, 0.7, Qinit=100,  es_kwargs={'epsilon':0})
ag_exp_m = Q_learner("more explore", env.actions, 0.7, es_kwargs={'epsilon':0.5})
ag_greedy = Q_learner("disc 0.1", env.actions, 0.1, Qinit=100)
sa = SARSA("SARSA", env.actions, 0.9)
sucb = SARSA("SARSA ucb", env.actions, 0.9, exploration_strategy = ucb, es_kwargs={'c':1})

sim_ag = Simulate(ag,env).start()

# sim_ag.go(1000)
# ag.Q    # get the learned Q-values
# sim_ag.plot()
# sim_ucb = Simulate(ag_ucb,env).start(); sim_ucb.go(1000); sim_ucb.plot()
# Simulate(ag_opt,env).start().go(1000).plot()
# Simulate(ag_exp_m,env).start().go(1000).plot()
# Simulate(ag_greedy,env).start().go(1000).plot()
# Simulate(sa,env).start().go(1000).plot()
# Simulate(sucb,env).start().go(1000).plot()

from mdpExamples import MDPtiny
envt = Env_from_ProblemDomain(MDPtiny())
agt = Q_learner("Q alpha=0.8", envt.actions, 0.8)
#Simulate(agt, envt).start().go(1000).plot()

##### Monster Game ####
mon_env = Monster_game_env()
mag1 = Q_learner("Q alpha=0.2", mon_env.actions, 0.9)
#Simulate(mag1,mon_env).start().go(100000).plot() 
mag_ucb = Q_learner("UCB(0.1),alpha=0.2", mon_env.actions, 0.9,
                        exploration_strategy = ucb, es_kwargs={'c':0.1})
#Simulate(mag_ucb,mon_env).start().go(100000).plot() 

mag2 = Q_learner("Q alpha=1/k", mon_env.actions, 0.9,
                     alpha_fun=lambda k:1/k)
#Simulate(mag2,mon_env).start().go(100000).plot()
mag3 = Q_learner("alpha=10/(9+k)", mon_env.actions, 0.9,
                     alpha_fun=lambda k:10/(9+k))
#Simulate(mag3,mon_env).start().go(100000).plot()

mag4 = Q_learner("ucb & alpha=10/(9+k)", mon_env.actions, 0.9,
                 alpha_fun=lambda k:10/(9+k),
                 exploration_strategy = ucb, es_kwargs={'c':0.1})
#Simulate(mag4,mon_env).start().go(100000).plot()

