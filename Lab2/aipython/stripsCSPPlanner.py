# stripsCSPPlanner.py - CSP planner where actions are represented using STRIPS
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from cspProblem import Variable, CSP, Constraint

class CSP_from_STRIPS(CSP):
    """A CSP where:
    * CSP variables are constructed for each feature and time, and each action and time
    * the dynamics are specified by the STRIPS representation of actions
    """

    def __init__(self,  planning_problem, number_stages=2):
        prob_domain = planning_problem.prob_domain
        initial_state = planning_problem.initial_state
        goal = planning_problem.goal
        # self.action_vars[t] is the action variable for time t
        self.action_vars = [Variable(f"Action{t}", prob_domain.actions)
                                for t in range(number_stages)]
        # feat_time_var[f][t] is the variable for feature f at time t
        feat_time_var = {feat: [Variable(f"{feat}_{t}",dom)
                                         for t in range(number_stages+1)]
                           for (feat,dom) in prob_domain.feature_domain_dict.items()}
                         
        # initial state constraints:
        constraints = [Constraint([feat_time_var[feat][0]], is_(val),
                                      f"{feat}[0]={val}")
                            for (feat,val) in initial_state.items()]
                            
        # goal constraints on the final state:
        constraints += [Constraint([feat_time_var[feat][number_stages]],
                                        is_(val), f"{feat}[{number_stages}]={val}")
                            for (feat,val) in goal.items()]
                            
        # precondition constraints:
        constraints += [Constraint([feat_time_var[feat][t], self.action_vars[t]],
                                   if_(val,act),
                                   f"{feat}[{t}]={val} if action[{t}]={act}")
                            for act in prob_domain.actions
                            for (feat,val) in act.preconds.items()
                            for t in range(number_stages)]
                            
        # effect constraints:
        constraints += [Constraint([feat_time_var[feat][t+1], self.action_vars[t]],
                                   if_(val,act),
                                   f"{feat}[{t+1}]={val} if action[{t}]={act}")
                            for act in prob_domain.actions
                            for feat,val in act.effects.items()
                            for t in range(number_stages)]
        # frame constraints:
        
        constraints += [Constraint([feat_time_var[feat][t], self.action_vars[t], feat_time_var[feat][t+1]],
                                   eq_if_not_in_({act for act in prob_domain.actions
                                                  if feat in act.effects}),
                                f"{feat}[t]={feat}[{t+1}] if act not in {set(act for act in prob_domain.actions if feat in act.effects)}")
                            for feat in prob_domain.feature_domain_dict
                            for t in range(number_stages) ]
        variables = set(self.action_vars) | {feat_time_var[feat][t]
                                            for feat in prob_domain.feature_domain_dict
                                            for t in range(number_stages+1)}
        CSP.__init__(self, "CSP_from_Strips", variables, constraints)

    def extract_plan(self,soln):
        return [soln[a] for a in self.action_vars]

def is_(val):
    """returns a function that is true when it is it applied to val.
    """
    #return lambda x: x == val
    def is_fun(x):
        return x == val
    is_fun.__name__ = f"value_is_{val}"
    return is_fun

def if_(v1,v2):
    """if the second argument is v2, the first argument must be v1"""
    #return lambda x1,x2: x1==v1 if x2==v2 else True
    def if_fun(x1,x2): 
        return x1==v1 if x2==v2 else True
    if_fun.__name__ = f"if x2 is {v2} then x1 is {v1}"
    return if_fun

def eq_if_not_in_(actset):
    """first and third arguments are equal if action is not in actset"""
    # return lambda x1, a, x2: x1==x2 if a not in actset else True
    def eq_if_not_fun(x1, a, x2):
        return x1==x2 if a not in actset else True
    eq_if_not_fun.__name__ = f"first and third arguments are equal if action is not in {actset}"
    return eq_if_not_fun

def con_plan(prob,horizon):
    """finds a plan for problem prob given horizon.
    """
    csp = CSP_from_STRIPS(prob, horizon)
    sol = Con_solver(csp).solve_one()
    return csp.extract_plan(sol) if sol else sol
    
from searchGeneric import Searcher
from cspConsistency import Search_with_AC_from_CSP, Con_solver
from stripsProblem import Planning_problem
import stripsProblem

# Problem 0
# con_plan(stripsProblem.problem0,1) # should it succeed?
# con_plan(stripsProblem.problem0,2) # should it succeed?
# con_plan(stripsProblem.problem0,3) # should it succeed?
# To use search to enumerate solutions
#searcher0a = Searcher(Search_with_AC_from_CSP(CSP_from_STRIPS(stripsProblem.problem0, 1)))
#print(searcher0a.search())  # returns path to solution

## Problem 1
# con_plan(stripsProblem.problem1,5) # should it succeed?
# con_plan(stripsProblem.problem1,4) # should it succeed?
## To use search to enumerate solutions:
#searcher15a = Searcher(Search_with_AC_from_CSP(CSP_from_STRIPS(stripsProblem.problem1, 5)))
#print(searcher15a.search())  # returns path to solution

## Problem 2
#con_plan(stripsProblem.problem2, 6)  # should fail??
#con_plan(stripsProblem.problem2, 7)  # should succeed???

## Example 6.13
problem3 = Planning_problem(stripsProblem.delivery_domain, 
                            {'SWC':True, 'RHC':False}, {'SWC':False})
#con_plan(problem3,2)  # Horizon of 2
#con_plan(problem3,3)  # Horizon of 3

problem4 = Planning_problem(stripsProblem.delivery_domain,{'SWC':True},
                               {'SWC':False, 'MW':False, 'RHM':False})

# For the stochastic local search:
#from cspSLS import SLSearcher, Runtime_distribution
# cspplanning15 = CSP_from_STRIPS(stripsProblem.problem1, 5) # should succeed
#se0 = SLSearcher(cspplanning15); print(se0.search(100000,0.5))
#p = Runtime_distribution(cspplanning15)
#p.plot_runs(1000,1000,0.7)  # warning may take a few minutes

