# cspProblem.py - Representations of a Constraint Satisfaction Problem
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from variable import Variable 

# for showing csps:
import matplotlib.pyplot as plt
import matplotlib.lines as lines

class Constraint(object):
    """A Constraint consists of
    * scope: a tuple or list of variables
    * condition: a Boolean function that can applied to a tuple of values for variables in scope
    * string: a string for printing the constraint
    """
    def __init__(self, scope, condition, string=None, position=None):
        self.scope = scope
        self.condition = condition
        self.string = string
        self.position = position

    def __repr__(self):
        return self.string

    def can_evaluate(self, assignment):
        """
        assignment is a variable:value dictionary
        returns True if the constraint can be evaluated given assignment
        """
        return all(v in assignment for v in self.scope)
    
    def holds(self,assignment):
        """returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) is true
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))

class CSP(object):
    """A CSP consists of
    * a title (a string)
    * variables, a list or set of variables
    * constraints, a list of constraints
    * var_to_const, a variable to set of constraints dictionary
    """
    def __init__(self, title, variables, constraints):
        """title is a string
        variables is set of variables
        constraints is a list of constraints
        """
        self.title = title
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var:set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """string representation of CSP"""
        return self.title

    def __repr__(self):
        """more detailed string representation of CSP"""
        return f"CSP({self.title}, {self.variables}, {([str(c) for c in self.constraints])})"

    def consistent(self,assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                    for con in self.constraints
                    if con.can_evaluate(assignment))

    def show(self, linewidth=3, showDomains=False, showAutoAC = False):
        self.linewidth = linewidth
        self.picked = None
        plt.ion()   # interactive
        self.arcs = {}   # arc: (con,var) dictionary
        self.thelines = {}  # (con,var):arc dictionary
        self.nodes = {}  # node: variable dictionary
        self.fig, self.ax= plt.subplots(1, 1)
        self.ax.set_axis_off()
        for var in self.variables:
            if var.position is None:
                var.position = (random.random(), random.random())
        self.showAutoAC = showAutoAC # used for consistency GUI
        self.autoAC = False
        domains = {var:var.domain for var in self.variables} if showDomains else {}
        self.draw_graph(domains=domains)

    def draw_graph(self, domains={}, to_do = {}, title=None, fontsize=10):
        self.ax.clear()
        self.ax.set_axis_off()
        if title:
            plt.title(title, fontsize=fontsize)
        else:
            plt.title(self.title, fontsize=fontsize)
        var_bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5",
                            facecolor="yellow")
        con_bbox = dict(boxstyle="square,pad=1.0",facecolor="lightyellow")
        self.autoACtext = plt.text(0,0,"Auto AC" if self.showAutoAC else "",
                                       bbox={'boxstyle':'square,pad=1.0',
                                                 'facecolor':'pink'},
                                       picker=True, fontsize=fontsize)
        for con in self.constraints:
            if con.position is None:
                con.position = tuple(sum(var.position[i] for var in con.scope)/len(con.scope)
                                         for i in range(2))
            cx,cy = con.position
            bbox = con_bbox
            for var in con.scope:
                vx,vy = var.position
                if (var,con) in to_do:
                    color = 'blue'
                else:
                    color = 'green'
                line = lines.Line2D([cx,vx], [cy,vy], axes=self.ax, color=color,
                                    picker=True, pickradius=10, linewidth=self.linewidth)
                self.arcs[line]= (var,con)
                self.thelines[(var,con)] = line
                self.ax.add_line(line)
            plt.text(cx,cy,con.string, 
                                   bbox=con_bbox,
                                   ha='center',va='center', fontsize=fontsize)
        for var in self.variables:
            x,y = var.position
            if domains:
                node_label = f"{var.name}\n{domains[var]}"
            else:
                node_label = var.name
            node = plt.text(x, y, node_label, bbox=var_bbox, ha='center', va='center',
                         picker=True, fontsize=fontsize)
            self.nodes[node] = var
        self.fig.canvas.mpl_connect('pick_event', self.pick_handler)

    def pick_handler(self,event):
        mouseevent = event.mouseevent
        self.last_artist = artist = event.artist
        #print('***picker handler:',artist, 'mouseevent:', mouseevent)
        if artist in self.arcs:
            #print('### selected arc',self.arcs[artist])
            self.picked = self.arcs[artist]
        elif artist in self.nodes:
            #print('### selected node',self.nodes[artist])
            self.picked = self.nodes[artist]
        elif artist==self.autoACtext:
            self.autoAC = True
            #print("*** autoAC")
        else:
            print("### unknown click")


