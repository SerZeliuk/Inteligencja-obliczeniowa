# mdpGUI.py - GUI for value iteration in MDPs
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox
from mdpProblem import MDP

class GridDomain(object):

    def viGUI(self):
        fig,self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        stepB = Button(plt.axes([0.8,0.05,0.1,0.075]), "step")
        stepB.on_clicked(self.on_step)
        resetB = Button(plt.axes([0.65,0.05,0.1,0.075]), "reset")
        resetB.on_clicked(self.on_reset)
        self.qcheck = CheckButtons(plt.axes([0.2,0.05,0.35,0.075]),
                                       ["show Q-values","show policy"])
        self.qcheck.on_clicked(self.show_vals)
        self.font_box = TextBox(plt.axes([0.1,0.05,0.05,0.075]),
                                    "Font:", textalignment="center")
        self.font_box.on_submit(self.set_font_size)
        self.font_box.set_val(str(plt.rcParams['font.size']))
        self.show_vals(None)
        plt.show()

    def set_font_size(self, s):
        plt.rcParams.update({'font.size': eval(s)})
        plt.draw()
        
    def show_vals(self,event):
        self.ax.cla() # clear the axes
        
        array = [[self.V[self.pos2state((x,y))] for x in range(self.x_dim)]
                                             for y in range(self.y_dim)]
        self.ax.pcolormesh([x-0.5  for x in range(self.x_dim+1)],
                               [y-0.5  for y in range(self.y_dim+1)],
                               array, edgecolors='black',cmap='summer')
            # for cmap see https://matplotlib.org/stable/tutorials/colors/colormaps.html
        if self.qcheck.get_status()[1]:  # "show policy"
                for x in range(self.x_dim):
                   for y in range(self.y_dim):
                      state = self.pos2state((x,y))
                      maxv = max(self.Q[state][a] for a in self.actions)
                      for a in self.actions:
                          if self.Q[state][a] == maxv:
                              # draw arrow in appropriate direction
                              xoff, yoff = self.offsets[a]
                              self.ax.arrow(x,y,xoff*2,yoff*2,
                                    color='red',width=0.05, head_width=0.2,
                                    length_includes_head=True)
        if self.qcheck.get_status()[0]:  # "show q-values"
           self.show_q(event)
        else:
           self.show_v(event)
        self.ax.set_xticks(range(self.x_dim))
        self.ax.set_xticklabels(range(self.x_dim))
        self.ax.set_yticks(range(self.y_dim))
        self.ax.set_yticklabels(range(self.y_dim))
        plt.draw()
        
    def on_step(self,event):
        self.step()
        self.show_vals(event)

    def step(self):
        """The default step is one step of value iteration"""
        self.vi(1)

    def show_v(self,event):
        """show values"""
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                state = self.pos2state((x,y))
                self.ax.text(x,y,"{val:.2f}".format(val=self.V[state]),ha='center')

    def show_q(self,event):
        """show q-values"""
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                state = self.pos2state((x,y))
                for a in self.actions:
                    xoff, yoff = self.offsets[a]
                    self.ax.text(x+xoff,y+yoff,
                                 "{val:.2f}".format(val=self.Q[state][a]),ha='center')

    def on_reset(self,event):
       self.V = {s:self.vinit for s in self.states}
       self.Q = {s: {a: self.vinit for a in self.actions} for s in self.states}
       self.show_vals(event)

# to use the GUI do some of:
import mdpExamples
# mdpExamples.MDPtiny(discount=0.9).viGUI()
# mdpExamples.grid(discount=0.9).viGUI()
# mdpExamples.Monster_game(discount=0.9).viGUI() # see mdpExamples.py

if __name__ == "__main__":
    print("Try:  mdpExamples.MDPtiny(discount=0.9).viGUI()")

