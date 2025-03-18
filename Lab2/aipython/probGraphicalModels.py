# probGraphicalModels.py - Graphical Models and Belief Networks
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
from variable import Variable
from probFactors import CPD, Prob
import matplotlib.pyplot as plt

class GraphicalModel(Displayable):
    """The class of graphical models. 
    A graphical model consists of a title, a set of variables and a set of factors.

    vars is a set of variables
    factors is a set of factors
    """
    def __init__(self, title, variables=None, factors=None):
        self.title = title
        self.variables = variables
        self.factors = factors

class BeliefNetwork(GraphicalModel):
    """The class of belief networks."""

    def __init__(self, title, variables, factors):
        """vars is a set of variables
        factors is a set of factors. All of the factors are instances of CPD (e.g., Prob).
        """
        GraphicalModel.__init__(self, title, variables, factors)
        assert all(isinstance(f,CPD) for f in factors), factors
        self.var2cpt = {f.child:f for f in factors}
        self.var2parents = {f.child:f.parents for f in factors}
        self.children = {n:[] for n in self.variables}
        for v in self.var2parents:
            for par in self.var2parents[v]:
                self.children[par].append(v)
        self.topological_sort_saved = None

    def topological_sort(self):
        """creates a topological ordering of variables such that the parents of 
        a node are before the node.
        """
        if self.topological_sort_saved:
            return self.topological_sort_saved
        next_vars = {n for n in self.var2parents if not self.var2parents[n] }
        self.display(3,'topological_sort: next_vars',next_vars)
        top_order=[]
        while next_vars:
            var = next_vars.pop()
            self.display(3,'select variable',var)
            top_order.append(var)
            next_vars |= {ch for ch in self.children[var]
                              if all(p in top_order for p in self.var2parents[ch])}
            self.display(3,'var_with_no_parents_left',next_vars)
        self.display(3,"top_order",top_order)
        assert set(top_order)==set(self.var2parents),(top_order,self.var2parents)
        self.topologicalsort_saved=top_order
        return top_order
    
    def show(self, fontsize=10, facecolor='orange'):
        plt.ion()   # interactive
        ax = plt.figure().gca()
        ax.set_axis_off()
        plt.title(self.title, fontsize=fontsize)
        bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5",facecolor=facecolor)
        for var in self.variables: #reversed(self.topological_sort()):
            for par in self.var2parents[var]:
                    ax.annotate(var.name, par.position, xytext=var.position,
                                    arrowprops={'arrowstyle':'<-'},bbox=bbox,
                                    ha='center', va='center', fontsize=fontsize)
        for var in self.variables:
                x,y = var.position
                plt.text(x,y,var.name,bbox=bbox,ha='center', va='center', fontsize=fontsize)

#### Simple Example Used for Unit Tests ####
boolean = [False, True]
A = Variable("A", boolean, position=(0,0.8))
B = Variable("B", boolean, position=(0.333,0.7))
C = Variable("C", boolean, position=(0.666,0.6))
D = Variable("D", boolean, position=(1,0.5))

f_a = Prob(A,[],[0.4,0.6])
f_b = Prob(B,[A],[[0.9,0.1],[0.2,0.8]])
f_c = Prob(C,[B],[[0.6,0.4],[0.3,0.7]])
f_d = Prob(D,[C],[[0.1,0.9],[0.75,0.25]])

bn_4ch = BeliefNetwork("4-chain", {A,B,C,D}, {f_a,f_b,f_c,f_d})

from display import Displayable

class InferenceMethod(Displayable):
    """The abstract class of graphical model inference methods"""
    method_name = "unnamed"  # each method should have a method name

    def __init__(self,gm=None):
        self.gm = gm

    def query(self, qvar, obs={}):
        """returns a {value:prob} dictionary for the query variable"""
        raise NotImplementedError("InferenceMethod query")   # abstract method

    def testIM(self, threshold=0.0000000001):
        solver = self(bn_4ch)
        res = solver.query(B,{D:True})
        correct_answer = 0.429632380245
        assert correct_answer-threshold < res[True] < correct_answer+threshold, \
                f"value {res[True]} not in desired range for {self.method_name}"
        print(f"Unit test passed for {self.method_name}.")
    
    def show_post(self, obs={}, num_format="{:.3f}", fontsize=10, facecolor='orange'):
        """draws the graphical model conditioned on observations obs
           num_format is number format (allows for more or less precision)
           fontsize gives size of the text
           facecolor gives the color of the nodes
        """
        plt.ion()   # interactive
        ax = plt.figure().gca()
        ax.set_axis_off()
        plt.title(self.gm.title+" observed: "+str(obs), fontsize=fontsize)
        bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5", facecolor=facecolor)
        vartext = {} # variable:text dictionary
        for var in self.gm.variables: #reversed(self.gm.topological_sort()):
            if var in obs:
                text =  var.name + "=" + str(obs[var])
            else:
                distn = self.query(var, obs=obs)
                
                text = var.name + "\n" + "\n".join(str(d)+": "+num_format.format(v) for (d,v) in distn.items())
            vartext[var] = text
            # Draw arcs 
            for par in self.gm.var2parents[var]:
                    ax.annotate(text, par.position, xytext=var.position,
                                    arrowprops={'arrowstyle':'<-'},bbox=bbox,
                                    ha='center', va='center', fontsize=fontsize)
        for var in self.gm.variables:
            x,y = var.position
            plt.text(x,y,vartext[var], bbox=bbox, ha='center', va='center', fontsize=fontsize)
                
