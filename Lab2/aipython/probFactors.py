# probFactors.py - Factors for graphical models
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
import math

class Factor(Displayable):
    nextid=0  # each factor has a unique identifier; for printing

    def __init__(self, variables, name=None):
        self.variables = variables   # list of variables
        if name:
            self.name = name
        else:
            self.name = f"f{Factor.nextid}"
            Factor.nextid += 1
        
    def can_evaluate(self,assignment):
        """True when the factor can be evaluated in the assignment
        assignment is a {variable:value} dict
        """
        return all(v in assignment for v in self.variables)
    
    def get_value(self,assignment):
        """Returns the value of the factor given the assignment of values to variables.
        Needs to be defined for each subclass.
        """
        assert self.can_evaluate(assignment)
        raise NotImplementedError("get_value")   # abstract method

    def __str__(self):
        """returns a string representing a summary of the factor"""
        return f"{self.name}({','.join(str(var) for var in self.variables)})"

    def to_table(self, variables=None, given={}):
        """returns a string representation of the factor.
        Allows for an arbitrary variable ordering.
        variables is a list of the variables in the factor
        (can contain other variables)"""
        if variables==None:
            variables = [v for v in self.variables if v not in given]
        else:  #enforce ordering and allow for extra variables in ordering
            variables = [v for v in variables if v in self.variables and v not in given]
        head = "\t".join(str(v) for v in variables)+"\t"+self.name
        return head+"\n"+self.ass_to_str(variables, given, variables)

    def ass_to_str(self, vars, asst, allvars):
        #print(f"ass_to_str({vars}, {asst}, {allvars})")
        if vars:
            return "\n".join(self.ass_to_str(vars[1:], {**asst, vars[0]:val}, allvars)
                            for val in vars[0].domain)
        else:
            val = self.get_value(asst)
            val_st =  "{:.6f}".format(val) if isinstance(val,float) else str(val)
            return ("\t".join(str(asst[var]) for var in allvars)
                        + "\t"+val_st)
        
    __repr__ = __str__
        
class CPD(Factor):
    def __init__(self, child, parents):
        """represents P(variable | parents)
        """
        self.parents = parents
        self.child = child
        Factor.__init__(self, parents+[child], name=f"Probability")

    def __str__(self):
        """A brief description of a factor using in tracing"""
        if self.parents:
            return f"P({self.child}|{','.join(str(p) for p in self.parents)})"
        else:
            return f"P({self.child})"
        
    __repr__ = __str__

class ConstantCPD(CPD):
    def __init__(self, variable, value):
        CPD.__init__(self, variable, [])
        self.value = value
    def get_value(self, assignment):
        return 1 if self.value==assignment[self.child] else 0
    
from learnLinear import sigmoid, logit

class LogisticRegression(CPD):
    def __init__(self, child, parents, weights):
        """A logistic regression representation of a conditional probability.
        child is the Boolean (or 0/1) variable whose CPD is being defined
        parents is the list of parents
        weights is list of parameters, such that weights[i+1] is the weight for parents[i]
          weights[0] is the bias.
        """
        assert len(weights) == 1+len(parents)
        CPD.__init__(self, child, parents)
        self.weights = weights

    def get_value(self,assignment):
        assert self.can_evaluate(assignment)
        prob = sigmoid(self.weights[0]
                        + sum(self.weights[i+1]*assignment[self.parents[i]]
                                  for i in range(len(self.parents))))
        if assignment[self.child]:  #child is true
            return prob
        else:
            return (1-prob)

class NoisyOR(CPD):
    def __init__(self, child, parents, weights):
        """A noisy representation of a conditional probability.
        variable is the Boolean (or 0/1) child variable whose CPD is being defined
        parents is the list of Boolean (or 0/1) parents
        weights is list of parameters, such that weights[i+1] is the weight for parents[i]
        """
        assert len(weights) == 1+len(parents)
        CPD.__init__(self, child, parents)
        self.weights = weights

    def get_value(self,assignment):
        assert self.can_evaluate(assignment)
        probfalse = (1-self.weights[0])*math.prod(1-self.weights[i+1]
                                             for i in range(len(self.parents))
                                                if assignment[self.parents[i]])
        if assignment[self.child]:  # child is assigned True in assignment
            return 1-probfalse
        else:
            return probfalse

class TabFactor(Factor):
    
    def __init__(self, variables, values, name=None):
        Factor.__init__(self, variables, name=name)
        self.values = values

    def get_value(self,  assignment):
        return self.get_val_rec(self.values, self.variables, assignment)
    
    def get_val_rec(self, value, variables, assignment):
        if variables == []:
           return value
        else:
            return self.get_val_rec(value[assignment[variables[0]]],
                                        variables[1:],assignment)

class Prob(CPD,TabFactor):
    """A factor defined by a conditional probability table"""
    def __init__(self, var, pars, cpt, name=None):
        """Creates a factor from a conditional probability table, cpt 
        The cpt values are assumed to be for the ordering par+[var]
        """
        TabFactor.__init__(self, pars+[var], cpt, name)
        self.child = var
        self.parents = pars

class ProbDT(CPD):
    def __init__(self, child, parents, dt):
        CPD.__init__(self, child, parents)
        self.dt = dt

    def get_value(self, assignment):
        return self.dt.get_value(assignment, self.child)

    def can_evaluate(self, assignment):
        return self.child in assignment and self.dt.can_evaluate(assignment)

class IFeq:
    def __init__(self, var, val, true_cond, false_cond):
        self.var = var
        self.val = val
        self.true_cond = true_cond
        self.false_cond = false_cond

    def get_value(self, assignment, child):
        """  IFeq(var, val, true_cond, false_cond) 
        value of true_cond is used if var has value val in assignment, 
        value of false_cond is used if var has a different value
        """
        if assignment[self.var] == self.val:
            return self.true_cond.get_value(assignment, child)
        else:
            return self.false_cond.get_value(assignment,child)

    def can_evaluate(self, assignment):
        if self.var not in assignment:
            return False
        elif assignment[self.var] == self.val:
            return self.true_cond.can_evaluate(assignment)
        else:
            return self.false_cond.can_evaluate(assignment)

class SameAs:
    def __init__(self, parent):
        """1 when child has same value as parent, otherwise 0"""
        self.parent = parent

    def get_value(self, assignment, child):
        return 1 if assignment[child]==assignment[self.parent] else 0

    def can_evaluate(self, assignment):
        return self.parent in assignment  

class Dist:
    def __init__(self, dist):
        """Dist is an array or dictionary indexed by value of current child"""
        self.dist = dist

    def get_value(self, assignment, child):
        return self.dist[assignment[child]]

    def can_evaluate(self, assignment):
        return True  

##### A decision tree representation Example 9.18 of AIFCA 3e
from variable import Variable

boolean = [False, True]

action = Variable('Action', ['go_out', 'get_coffee'], position=(0.5,0.8))
rain = Variable('Rain', boolean, position=(0.2,0.8))
full = Variable('Cup Full', boolean, position=(0.8,0.8))

wet = Variable('Wet', boolean, position=(0.5,0.2))
p_wet = ProbDT(wet,[action,rain,full],
                   IFeq(action, 'go_out',
                        IFeq(rain, True, Dist([0.2,0.8]), Dist([0.9,0.1])),
                        IFeq(full, True, Dist([0.4,0.6]), Dist([0.7,0.3]))))

# See probRC for wetBN which expands this example to a complete network

class FactorObserved(Factor):
    def __init__(self,factor,obs):
        Factor.__init__(self, [v for v in factor.variables if v not in obs])
        self.observed = obs
        self.orig_factor = factor

    def get_value(self,assignment):
        return self.orig_factor.get_value(assignment|self.observed)

class FactorSum(Factor):
    def __init__(self,var,factors):
        self.var_summed_out = var
        self.factors = factors
        vars = list({v for fac in factors
                       for v in fac.variables if v is not var})
        #for fac in factors:
        #    for v in fac.variables:
        #        if v is not var and v not in vars:
        #            vars.append(v)
        Factor.__init__(self,vars)
        self.values = {}

    def get_value(self,assignment):
        """lazy implementation: if not saved, compute it. Return saved value"""
        asst = frozenset(assignment.items())
        if asst in self.values:
            return self.values[asst]
        else:
            total = 0
            new_asst = assignment.copy()
            for val in self.var_summed_out.domain:
                new_asst[self.var_summed_out] = val
                total += math.prod(fac.get_value(new_asst) for fac in self.factors)
            self.values[asst] = total
            return total

def factor_times(variable, factors):
    """when factors are factors just on variable (or on no variables)"""
    prods = []
    facs = [f for f in factors if variable in f.variables]
    for val in variable.domain:
        ast = {variable:val}
        prods.append(math.prod(f.get_value(ast) for f in facs))
    return prods
    
