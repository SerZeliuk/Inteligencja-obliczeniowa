# relnProbModels.py - Relational Probabilistic Models: belief networks with plates
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
from probGraphicalModels import BeliefNetwork
from variable import Variable
from probRC import ProbRC
from probFactors import Prob
import random

boolean = [False, True]

class ParVar(object):
    """Parametrized random variable"""
    def __init__(self, name, log_vars, domain, position=None):
        self.name = name # string
        self.log_vars = log_vars
        self.domain = domain # list of values
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain)

class RBN(Displayable):
    def __init__(self, title, parvars, parfactors):
        self.title = title
        self.parvars = parvars
        self.parfactors = parfactors
        self.log_vars = {V for PV in parvars for V in PV.log_vars}

    def ground(self, populations, offsets=None):
        """Ground the belief network with the populations of the logical variables.
        populations is a dictionary that maps each logical variable to the list of individuals.
        Returns a belief network representation of the grounding.
        """
        assert all(lv in populations for lv in self.log_vars), f"{[lv for lv in self.log_vars if lv not in populations]} have no population"
        self.cps = []     # conditional probabilities in the grounding
        self.var_dict = {}     # ground variables created
        for pp in self.parfactors:
            self.ground_parfactor(pp, list(self.log_vars), populations, {}, offsets)
        return BeliefNetwork(self.title+"_grounded", self.var_dict.values(), self.cps)

    def ground_parfactor(self, parfactor, lvs, populations, context, offsets):
        """
        parfactor is the parfactor to get instances of
        lvs is a list of the logical variables in parfactor not assigned in context
        populations is {logical_variable: population} dictionary
        context is a {logical_variable:value} dictionary for logical_variable in parfactor
        offsets a {loc_var:(x_offset,y_offset)} dictionary or None
        """
        if lvs == []:
            if isinstance(parfactor, Prob):
                self.cps.append(Prob(self.ground_pvr(parfactor.child,context,offsets),
                                         [self.ground_pvr(p,context,offsets) for p in parfactor.parents],
                                         parfactor.values))
            else:
                print("Parfactor not implemented for",parfactor,"of type",type(parfactor))
        else:
            for val in populations[lvs[0]]:
                self.ground_parfactor(parfactor, lvs[1:], populations, {lvs[0]:val}|context, offsets)

    def ground_pvr(self, prv, context, offsets):
        """grounds a parametrized random variable with respect to a context
        prv is a parametrized random variable
        context is a logical_variable:value dictionary that assigns all logical variables in prv 
        offsets a {loc_var:(x_offset,y_offset)} dictionary or None
        """
        if isinstance(prv,ParVar):
            args = tuple(context[lv] for lv in prv.log_vars)
            if (prv,args) in self.var_dict:
                return self.var_dict[(prv,args)]
            else:
                new_gv = GrVar(prv, args, offsets)
                self.var_dict[(prv,args)] = new_gv
                return new_gv
        else:  # allows for non-parametrized random variables
            return prv

class GrVar(Variable):
    """Grounded Variable"""
    def __init__(self, parvar, args, offsets = None):
        """A grounded variable 
        parvar is the parametrized variable
        args is a tuple of a value for each random variable
        offsets is a map between the value and the (x,y) offsets
        """
        if offsets:
            pos = sum_positions([parvar.position]+[offsets[a] for a in args])
        else:
           pos = sum_positions([parvar.position, (random.uniform(-0.2,0.2),random.uniform(-0.2,0.2))])
        Variable.__init__(self,parvar.name+"("+",".join(args)+")", parvar.domain, pos)
        self.parvar= parvar
        self.args = tuple(args)
        self.hash_value = None

    def __hash__(self):
        if self.hash_value is None:  # only hash once
            self.hash_value = hash((self.parvar, self.args))
        return self.hash_value

    def __eq__(self, other):
        return isinstance(other,GrVar) and self.parvar == other.parvar and self.args == other.args

def sum_positions(poslist):
    (x,y) = (0,0)
    for (xo,yo) in poslist:
        x += xo
        y += yo
    return (x,y)

Int = ParVar("Intelligent", ["St"], boolean, position=(0.0,0.7))
Grade = ParVar("Grade", ["St","Co"], ["A", "B", "C"], position=(0.2,0.6))
Diff = ParVar("Difficult", ["Co"], boolean, position=(0.3,0.9))

pg = Prob(Grade, [Int, Diff],
                [[{"A": 0.1, "B":0.4, "C":0.5},
                      {"A": 0.01, "B":0.09, "C":0.9}],
                 [{"A": 0.9, "B":0.09, "C":0.01},
                       {"A": 0.5, "B":0.4, "C":0.1}]])
pi = Prob( Int, [], [0.5, 0.5])
pd = Prob( Diff, [], [0.5, 0.5])
grades = RBN("Grades RBN", {Int, Grade, Diff}, {pg,pi,pd})

students = ["s1", "s2", "s3", "s4"]
st_offsets = {st:(0,-0.2*i) for (i,st) in enumerate(students)}
courses = ["c1", "c2", "c3", "c4"]
co_offsets = {co:(0.2*i,0) for (i,co) in enumerate(courses)}
grades_gr = grades.ground({"St": students, "Co": courses},
                            offsets= st_offsets | co_offsets)

obs = {GrVar(Grade,["s1","c1"]):"A", GrVar(Grade,["s2","c1"]):"C", GrVar(Grade,["s1","c2"]):"B",
           GrVar(Grade,["s2","c3"]):"B", GrVar(Grade,["s3","c2"]):"B", GrVar(Grade,["s4","c3"]):"B"}

# grades_rc = ProbRC(grades_gr)
# grades_rc.show_post({GrVar(Grade,["s1","c1"]):"A"},fontsize=10)
# grades_rc.show_post({GrVar(Grade,["s1","c1"]):"A",GrVar(Grade,["s2","c1"]):"C"})
# grades_rc.show_post({GrVar(Grade,["s1","c1"]):"A",GrVar(Grade,["s2","c1"]):"C", GrVar(Grade,["s1","c2"]):"B"})
# grades_rc.show_post(obs,fontsize=10)
# grades_rc.query(GrVar(Grade,["s3","c4"]), obs)
# grades_rc.query(GrVar(Grade,["s4","c4"]), obs)
# grades_rc.query(GrVar(Int,["s3"]), obs)  
# grades_rc.query(GrVar(Int,["s4"]), obs)  

