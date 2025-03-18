# mdpExamples.py - MDP Examples
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from mdpProblem import MDP, ProblemDomain, distribution
from mdpGUI import GridDomain
import matplotlib.pyplot as plt

class partyMDP(MDP): 
    """Simple 2-state, 2-Action Partying MDP Example"""
    def __init__(self, discount=0.9):
        states = {'healthy','sick'}
        actions = {'relax', 'party'}
        MDP.__init__(self, "party MDP", states, actions, discount)

    def R(self,s,a):
        "R(s,a)"
        return { 'healthy': {'relax': 7, 'party': 10},
                 'sick':    {'relax': 0, 'party': 2 }}[s][a]

    def P(self,s,a):
        "returns a dictionary of {s1:p1} such that P(s1 | s,a)=p1. Other probabilities are zero."
        phealthy = {  # P('healthy' | s, a)
                     'healthy': {'relax': 0.95, 'party': 0.7},
                     'sick': {'relax': 0.5, 'party': 0.1 }}[s][a]
        return {'healthy':phealthy, 'sick':1-phealthy}

class MDPtiny(ProblemDomain, GridDomain):
    def __init__(self, discount=0.9):
        x_dim = 2   # x-dimension
        y_dim = 3
        ProblemDomain.__init__(self,
            "Tiny MDP", # title
            [(x,y) for x in range(x_dim) for y in range(y_dim)], #states
            ['right', 'upC', 'left', 'upR'], #actions
            discount,
            x_dim=x_dim, y_dim = y_dim,
            offsets = {'right':(0.25,0), 'upC':(0,-0.25), 'left':(-0.25,0), 'upR':(0,0.25)}
            )

    def result(self, state, action):
        """return a dictionary of {(r,s):p} where p is the probability of reward r, state s
        a state is an (x,y) pair
        """
        (x,y) = state
        right = (-x,(1,y)) # reward is -1 if x was 1
        left =  (0,(0,y)) if x==1 else [(-1,(0,0)), (-100,(0,1)), (10,(0,0))][y]
        up = (0,(x,y+1)) if y<2 else (-1,(x,y))
        if action == 'right':
            return {right:1}
        elif action == 'upC':
            (r,s) = up
            return {(r-1,s):1}
        elif action == 'left':
           return {left:1}
        elif action == 'upR':
            return distribution({left: 0.1}).add_prob(right,0.1).add_prob(up,0.8)
            # Exercise: what is wrong with return {left: 0.1, right:0.1, up:0.8}

# To show GUI do
#  MDPtiny().viGUI()

class grid(ProblemDomain, GridDomain):
    """ x_dim * y_dim grid with rewarding states"""
    def __init__(self, discount=0.9, x_dim=10, y_dim=10):
        ProblemDomain.__init__(self,
            "Grid World",
            [(x,y) for x in range(y_dim) for y in range(y_dim)], #states
            ['up', 'down', 'right', 'left'], #actions
            discount,
            x_dim = x_dim, y_dim = y_dim,
            offsets = {'right':(0.25,0), 'up':(0,0.25), 'left':(-0.25,0), 'down':(0,-0.25)})
        self.rewarding_states = {(3,2):-10, (3,5):-5, (8,2):10, (7,7):3 }
        self.fling_states = {(8,2), (7,7)}  # assumed a subset of rewarding_states
 
    def intended_next(self,s,a):
        """returns the (reward, state)  in the direction a.
        This is where the agent will end up if to goes in its intended_direction
             (which it does with probability 0.7).
        """
        (x,y) = s
        if a=='up':
            return (0, (x,y+1)) if y+1 < self.y_dim else (-1, (x,y))
        if a=='down':
            return (0, (x,y-1)) if y > 0 else (-1, (x,y))
        if a=='right':
            return (0, (x+1,y)) if x+1 < self.x_dim else (-1, (x,y))
        if a=='left':
            return (0, (x-1,y)) if x > 0 else (-1, (x,y))

    def result(self,s,a):
        """return a dictionary of {(r,s):p} where p is the probability of reward r, state s.
        a state is an (x,y) pair
        """
        r0 = self.rewarding_states[s] if s in self.rewarding_states else 0
        if s in self.fling_states:
            return {(r0,(0,0)): 0.25, (r0,(self.x_dim-1,0)):0.25,
                        (r0,(0,self.y_dim-1)):0.25, (r0,(self.x_dim-1,self.y_dim-1)):0.25}
        dist = distribution({})
        for a1 in self.actions:
            (r1,s1) = self.intended_next(s,a1)
            rs = (r1+r0, s1)
            p = 0.7 if a1==a else 0.1
            dist.add_prob(rs,p)
        return dist

class Monster_game(ProblemDomain, GridDomain):

    vwalls = [(0,3), (0,4), (1,4)]  # vertical walls right of these locations
    crash_reward = -1
    
    prize_locs = [(0,0), (0,4), (4,0), (4,4)]
    prize_apears_prob = 0.3
    prize_reward = 10

    monster_locs = [(0,1), (1,1), (2,3), (3,1), (4,2)]
    monster_appears_prob = 0.4
    monster_reward_when_damaged = -10
    repair_stations = [(1,4)]

    def __init__(self, discount=0.9):
        x_dim = 5
        y_dim = 5
            # which damaged and prize to show
        ProblemDomain.__init__(self,
            "Monster Game",
            [(x,y,damaged,prize)
                 for x in range(x_dim)
                 for y in range(y_dim)
                 for damaged in [False,True]
                 for prize in [None]+self.prize_locs], #states
            ['up', 'down', 'right', 'left'], #actions
            discount,
            x_dim = x_dim, y_dim = y_dim,
            offsets = {'right':(0.25,0), 'up':(0,0.25), 'left':(-0.25,0), 'down':(0,-0.25)})
        self.state = (2,2,False,None)
        
    def intended_next(self,xy,a):
        """returns the (reward, (x,y))  in the direction a.
        This is where the agent will end up if to goes in its intended_direction
             (which it does with probability 0.7).
        """
        (x,y) = xy # original x-y position
        if a=='up':
            return (0, (x,y+1)) if y+1 < self.y_dim else (self.crash_reward, (x,y))
        if a=='down':
            return (0, (x,y-1)) if y > 0 else (self.crash_reward, (x,y))
        if a=='right':
            if (x,y) in self.vwalls or x+1==self.x_dim: # hit wall
                return (self.crash_reward, (x,y))
            else:
                return (0, (x+1,y)) 
        if a=='left':
            if (x-1,y) in self.vwalls or x==0: # hit wall
                            return (self.crash_reward, (x,y))
            else:
                return (0, (x-1,y)) 

    def result(self,s,a):
        """return a dictionary of {(r,s):p} where p is the probability of reward r, state s.
        a state is an (x,y) pair
        """
        (x,y,damaged,prize) = s
        dist = distribution({})
        for a1 in self.actions: # possible results
            mp = 0.7 if a1==a else 0.1
            mr,(xn,yn) = self.intended_next((x,y),a1)
            if (xn,yn) in self.monster_locs:
                if damaged:
                    dist.add_prob((mr+self.monster_reward_when_damaged,(xn,yn,True,prize)), mp*self.monster_appears_prob)
                    dist.add_prob((mr,(xn,yn,True,prize)), mp*(1-self.monster_appears_prob))
                else:
                   dist.add_prob((mr,(xn,yn,True,prize)), mp*self.monster_appears_prob)
                   dist.add_prob((mr,(xn,yn,False,prize)), mp*(1-self.monster_appears_prob))
            elif (xn,yn) == prize:
                dist.add_prob((mr+self.prize_reward,(xn,yn,damaged,None)), mp)
            elif (xn,yn) in self.repair_stations:
                dist.add_prob((mr,(xn,yn,False,prize)), mp)
            else:
                dist.add_prob((mr,(xn,yn,damaged,prize)), mp)
        if prize is None:
            res = distribution({})
            for (r,(x2,y2,d,p2)),p in dist.items():
                res.add_prob((r,(x2,y2,d,None)), p*(1-self.prize_apears_prob))
                for pz in self.prize_locs:
                    res.add_prob((r,(x2,y2,d,pz)), p*self.prize_apears_prob/len(self.prize_locs))
            return res
        else:
            return dist
            
    def state2pos(self, state):
        """When displaying as a grid, this specifies how the state is mapped to (x,y) position.
        The default is for domains where the (x,y) position is the state
        """
        (x,y,d,p) = state
        return (x,y)
        
    def pos2state(self, pos):
        """When displaying as a grid, this specifies how the state is mapped to (x,y) position.
        """
        (x,y) = pos
        (xs, ys, damaged, prize) = self.state
        return (x, y, damaged, prize)
        
    def state2goal(self,state):
        """the (x,y) position for the goal
        """
        (x, y, damaged, prize) = state
        return prize
        
# value iteration GUI for Monster game:
# mg = Monster_game()
# mg.viGUI()  # then run vi a few times
# to see other states, exit the GUI
# mg.state = (2,2,True,(4,4)) # or other damaged/prize states
# mg.viGUI()

## Testing value iteration
# Try the following:
# pt = partyMDP(discount=0.9)
# pt.vi(1)
# pt.vi(100)
# partyMDP(discount=0.99).vi(100)
# partyMDP(discount=0.4).vi(100)

# gr = grid(discount=0.9)
# gr.viGUI()
# q,v,pi = gr.vi(100)
# q[(7,2)]


## Testing asynchronous value iteration
# Try the following:
# pt = partyMDP(discount=0.9)
# pt.avi(10)
# pt.vi(1000)

# gr = grid(discount=0.9)
# q = gr.avi(100000)
# q[(7,2)]

def test_MDP(mdp, discount=0.9, eps=0.01):
    """tests vi and avi give the same answer for a MDP class mdp
    """
    mdp1 = mdp(discount=discount)
    q1,v1,pi1 = mdp1.vi(100)
    mdp2 = mdp(discount=discount)
    q2 = mdp2.avi(1000)
    same = all(abs(q1[s][a]-q2[s][a]) < eps
                   for s in mdp1.states
                   for a in mdp1.actions)
    assert same, "vi and avi are different:\n{q1}\n{q2}"
    print(f"passed unit test.  vi and avi gave same result for {mdp1.title}")
    
if __name__ == "__main__":
    test_MDP(partyMDP)
    
