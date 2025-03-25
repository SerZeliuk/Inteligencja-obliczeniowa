# agentTop.py - Top Layer
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable 
from agentMiddle import Rob_middle_layer
from agents import Environment

class Rob_top_layer(Environment):
    def __init__(self, middle, timeout=200, locations = {'mail':(-5,10), 
                          'o103':(50,10), 'o109':(100,10),'storage':(101,51)} ):
        """middle is the middle layer
        timeout is the number of steps the middle layer goes before giving up
        locations is a loc:pos dictionary 
            where loc is a named location, and pos is an (x,y) position.
        """
        self.middle = middle
        self.timeout = timeout  # number of steps before the middle layer should give up
        self.locations = locations
        
    def do(self,plan):
        """carry out actions.
        actions is of the form {'visit':list_of_locations}
        It visits the locations in turn.
        """
        to_do = plan['visit']
        for loc in to_do:
            position = self.locations[loc]
            arrived = self.middle.do({'go_to':position, 'timeout':self.timeout})
            self.display(1,"Goal",loc,arrived)

import matplotlib.pyplot as plt

class Plot_env(Displayable):
    def __init__(self, body,top):
        """sets up the plot
        """
        self.body = body
        self.top = top
        plt.ion()
        plt.axes().set_aspect('equal')
        self.redraw()

    def redraw(self):
        plt.clf()
        for wall in self.body.world.walls:
            ((x0,y0),(x1,y1)) = wall
            plt.plot([x0,x1],[y0,y1],"-k",linewidth=3)
        for loc in self.top.locations:
            (x,y) = self.top.locations[loc]
            plt.plot([x],[y],"k<")
            plt.text(x+1.0,y+0.5,loc) # print the label above and to the right
        plt.plot([self.body.rob_x],[self.body.rob_y],"go")
        plt.gca().figure.canvas.draw()
        if self.body.history or self.body.wall_history:
            self.plot_run()

    def plot_run(self):
        """plots the history after the agent has finished.
        This is typically only used if body.plotting==False
        """
        if self.body.history:
            xs,ys = zip(*self.body.history)
            plt.plot(xs,ys,"go")
        if self.body.wall_history:
            wxs,wys = zip(*self.body.wall_history)
            plt.plot(wxs,wys,"ro")

from agentEnv import Rob_body, Rob_world

world = Rob_world({((20,0),(30,20)), ((70,-5),(70,25))})
body = Rob_body(world)
middle = Rob_middle_layer(body)
top = Rob_top_layer(middle)

# try:
# pl=Plot_env(body,top)
# top.do({'visit':['o109','storage','o109','o103']})
# You can directly control the middle layer:
# middle.do({'go_to':(30,-10), 'timeout':200})
# Can you make it crash?

if __name__ == "__main__":
    print("Try: Plot_env(body,top); top.do({'visit':['o109','storage','o109','o103']})")

# Robot Trap for which the current controller cannot escape:
trap_env = Rob_world({((10,-21),(10,0)), ((10,10),(10,31)),
                          ((30,-10),(30,0)), ((30,10),(30,20)),
                          ((50,-21),(50,31)), ((10,-21),(50,-21)),
                          ((10,0),(30,0)),  ((10,10),(30,10)),
                          ((10,31),(50,31))})
trap_body = Rob_body(trap_env,init_pos=(-1,0,90))
trap_middle = Rob_middle_layer(trap_body)
trap_top = Rob_top_layer(trap_middle,locations={'goal':(71,0)})

# Robot trap exercise:
# pl=Plot_env(trap_body,trap_top)
# trap_top.do({'visit':['goal']})

