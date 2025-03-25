# rlGUI.py - Reinforcement Learning GUI
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox
from rlProblem import Simulate
        
class rlGUI(object):
    def __init__(self, env, agent):
        """
        """
        self.env = env
        self.agent = agent
        self.state = self.env.state
        self.x_dim = env.x_dim
        self.y_dim = env.y_dim
        if 'offsets' in vars(env):  # 'offsets' is defined in environment
            self.offsets = env.offsets
        else: # should be more general
            self.offsets = {'right':(0.25,0), 'up':(0,0.25), 'left':(-0.25,0), 'down':(0,-0.25)}
        # replace the exploration strategy with GUI
        self.orig_exp_strategy = self.agent.exploration_strategy
        self.agent.exploration_strategy = self.actionFromGUI
        self.do_steps = 0
        self.quitting = False
        self.action = None

    def go(self):
        self.q = self.agent.q
        self.v = self.agent.v
        try:
            self.fig,self.ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)
            self.actButtons = {self.fig.text(0.8+self.offsets[a][0]*0.4,0.1+self.offsets[a][1]*0.1,a,
                                    bbox={'boxstyle':'square','color':'yellow','ec':'black'},
                                    picker=True):a #, fontsize=fontsize):a
                 for a in self.env.actions}
            self.fig.canvas.mpl_connect('pick_event', self.sel_action)
            self.fig.canvas.mpl_connect('close_event', self.window_closed)
            self.sim = Simulate(self.agent, self.env)
            self.show()
            self.sim.start()
            self.sim.go(1000000000000) # go forever
        except ExitToPython:
            print("Window closed")

    def show(self):
        self.qcheck = CheckButtons(plt.axes([0.2,0.05,0.25,0.075]),
                                       ["show q-values","show policy","show visits"])
        self.qcheck.on_clicked(self.show_vals)
        self.font_box = TextBox(plt.axes([0.125,0.05,0.05,0.05]),"Font:", textalignment="center")
        self.font_box.on_submit(self.set_font_size)
        self.font_box.set_val(str(plt.rcParams['font.size']))
        self.step_box = TextBox(plt.axes([0.5,0.05,0.1,0.05]),"", textalignment="center")
        self.step_box.set_val("100")
        self.stepsButton = Button(plt.axes([0.6,0.05,0.075,0.05]), "steps", color='yellow')
        self.stepsButton.on_clicked(self.steps)
        #self.exitButton = Button(plt.axes([0.0,0.05,0.05,0.05]), "exit", color='yellow')
        #self.exitButton.on_clicked(self.exit)
        self.show_vals(None)

    def set_font_size(self, s):
        plt.rcParams.update({'font.size': eval(s)})
        plt.draw()

    def window_closed(self, s):
        self.quitting = True
        
    def show_vals(self,event):
        self.ax.cla()
        self.ax.set_title(f"{self.sim.step}: State: {self.state} Reward: {self.env.reward} Sum rewards: {self.sim.sum_rewards}")
        array = [[self.v(self.env.pos2state((x,y))) for x in range(self.x_dim)]
                                             for y in range(self.y_dim)]
        self.ax.pcolormesh([x-0.5  for x in range(self.x_dim+1)],
                               [x-0.5  for x in range(self.y_dim+1)],
                               array, edgecolors='black',cmap='summer')
            # for cmap see https://matplotlib.org/stable/tutorials/colors/colormaps.html
        if self.qcheck.get_status()[1]:  # "show policy"
                for x in range(self.x_dim):
                    for y in range(self.y_dim):
                       state = self.env.pos2state((x,y))
                       maxv = max(self.agent.q(state,a) for a in self.env.actions)
                       for a in self.env.actions:
                           xoff, yoff = self.offsets[a]
                           if self.agent.q(state,a) == maxv:
                              # draw arrow in appropriate direction
                              self.ax.arrow(x,y,xoff*2,yoff*2,
                                    color='red',width=0.05, head_width=0.2, length_includes_head=True)
        
        if goal := self.env.state2goal(self.state):
            self.ax.add_patch(plt.Circle(goal, 0.1, color='lime'))
        self.ax.add_patch(plt.Circle(self.env.state2pos(self.state), 0.1, color='w'))
        if self.qcheck.get_status()[0]:  # "show q-values"
           self.show_q(event)
        elif self.qcheck.get_status()[2] and 'visits' in vars(self.agent):  # "show visits"
           self.show_visits(event)
        else:
           self.show_v(event)
        self.ax.set_xticks(range(self.x_dim))
        self.ax.set_xticklabels(range(self.x_dim))
        self.ax.set_yticks(range(self.y_dim))
        self.ax.set_yticklabels(range(self.y_dim))
        plt.draw()
        
    def sel_action(self,event):
        self.action = self.actButtons[event.artist]

    def show_v(self,event):
        """show values"""
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                state = self.env.pos2state((x,y))
                self.ax.text(x,y,"{val:.2f}".format(val=self.agent.v(state)),ha='center')

    def show_q(self,event):
        """show q-values"""
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                state = self.env.pos2state((x,y))
                for a in self.env.actions:
                    xoff, yoff = self.offsets[a]
                    self.ax.text(x+xoff,y+yoff,
                                 "{val:.2f}".format(val=self.agent.q(state,a)),ha='center')

    def show_visits(self,event):
        """show q-values"""
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                state = self.env.pos2state((x,y))
                for a in self.env.actions:
                    xoff, yoff = self.offsets[a]
                    if state in self.agent.visits and a in self.agent.visits[state]:
                        num_visits = self.agent.visits[state][a]
                    else:
                        num_visits = 0
                    self.ax.text(x+xoff,y+yoff,
                                 str(num_visits),ha='center')
                                 
    def steps(self,event):
        "do the steps given in step box"
        num_steps = int(self.step_box.text)
        if num_steps > 0:
            self.do_steps = num_steps-1
            self.action = self.action_from_orig_exp_strategy()

    def action_from_orig_exp_strategy(self):
        """returns the action from the original explorations strategy"""
        visits = self.agent.visits[self.state] if 'visits' in vars(self.agent) else {}
        return self.orig_exp_strategy(self.state,{a:self.agent.q(self.state,a) for a in self.agent.actions},
                                     visits,**self.agent.es_kwargs)
        
    def actionFromGUI(self, state, *args, **kwargs):
        """called as the exploration strategy by the RL agent. 
        returns an action, either from the GUI or the original exploration strategy
        """
        self.state = state
        if self.do_steps > 0:  # use the original
            self.do_steps -= 1
            return self.action_from_orig_exp_strategy()
        else:  # get action from the user
            self.show_vals(None)
            while self.action == None and not self.quitting: #wait for user action
                plt.pause(0.05) # controls reaction time of GUI
            if self.quitting:
                raise ExitToPython()
            act = self.action
            self.action = None
            return act

class ExitToPython(Exception):
    """Thrown when window closes.
    """
    pass

from rlExamples import Monster_game_env
from mdpExamples import MDPtiny,  Monster_game
from rlQLearner import Q_learner, SARSA
from rlStochasticPolicy import StochasticPIAgent
from rlProblem import Env_from_ProblemDomain, epsilon_greedy, ucb

# Choose an Environment
env = Env_from_ProblemDomain(MDPtiny())
# env = Env_from_ProblemDomain(Monster_game())
# env = Monster_game_env()

# Choose an algorithm
# gui = rlGUI(env, Q_learner("Q", env.actions, 0.9)); gui.go()
# gui = rlGUI(env, SARSA("SARSA", env.actions, 0.9)); gui.go()
# gui = rlGUI(env, SARSA("SARSA alpha(k)=k:10/(9+k))", env.actions, 0.9, alpha_fun=lambda k:10/(9+k))); gui.go()
# gui = rlGUI(env, SARSA("SARSA-UCB", env.actions, 0.9, exploration_strategy = ucb, es_kwargs={'c':0.1})); gui.go()
# gui = rlGUI(env, StochasticPIAgent("Q", env.actions, 0.9, alpha_fun=lambda k:10/(9+k))); gui.go()

if __name__ == "__main__":
    print("Try: rlGUI(env, Q_learner('Q', env.actions, 0.9)).go()")
    
