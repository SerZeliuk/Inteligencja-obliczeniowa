# agentFollowTarget.py - Plotting for moving targets
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
from agentTop import Plot_env, body, top

class Plot_follow(Plot_env):
    def __init__(self, body, top, epsilon=2.5):
        """plot the agent in the environment. 
        epsilon is the threshold how how close someone needs to click to select a location.
        """
        Plot_env.__init__(self, body, top)
        self.epsilon = epsilon
        self.canvas = plt.gca().figure.canvas
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.pressloc = None
        self.pressevent = None
        for loc in self.top.locations:
            self.display(2,f"    loc {loc} at {self.top.locations[loc]}")

    def on_press(self, event):
        self.display(2,'v',end="")
        self.display(2,f"Press at ({event.xdata},{event.ydata}")
        for loc in self.top.locations:
            lx,ly = self.top.locations[loc]
            if abs(event.xdata- lx) <= self.epsilon and abs(event.ydata- ly) <= self.epsilon :
                self.pressloc = loc
                self.pressevent = event
                self.display(2,"moving",loc)

    def on_release(self, event):
        self.display(2,'^',end="")
        if self.pressloc is not None: #and event.inaxes == self.pressevent.inaxes:
            self.top.locations[self.pressloc] = (event.xdata, event.ydata)
            self.display(1,f"Placing {self.pressloc} at {(event.xdata, event.ydata)}")
        self.pressloc = None
        self.pressevent = None

    def on_move(self, event):
        if self.pressloc is not None: # and event.inaxes == self.pressevent.inaxes:
            self.display(2,'-',end="")
            self.top.locations[self.pressloc] = (event.xdata, event.ydata)
            self.redraw()
        else:
            self.display(2,'.',end="")

# try:
# pl=Plot_follow(body,top)
# top.do({'visit':['o109','storage','o109','o103']})

if __name__ == "__main__":
    print("Try: Plot_follow(body,top); top.do({'visit':['o109','storage','o109','o103']})")

