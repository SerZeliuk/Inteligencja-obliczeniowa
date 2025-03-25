# logicRelation.py - Datalog and Logic Programs
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
import logicProblem

class Var(Displayable):
    """A logical variable"""
    def __init__(self, name):
        """name"""
        self.name = name

    def __str__(self):
        return self.name
    __repr__ = __str__
    
    def __eq__(self, other):
        return isinstance(other,Var) and self.name == other.name
    def __hash__(self):
        return hash(self.name)

class Atom(object):
    """An atom"""
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return f"{self.name}({', '.join(str(a) for a in self.args)})"
    __repr__ = __str__

Func = Atom  # same syntax is used for function symbols

class Clause(logicProblem.Clause):
    next_index=0
    def __init__(self, head, *args, **nargs):
        if not isinstance(head, Atom):
            head = Atom(head)
        logicProblem.Clause.__init__(self, head, *args, **nargs)
        self.logical_variables = log_vars([self.head,self.body],set())

    def rename(self):
        """create a unique copy of the clause"""
        if self.logical_variables:
            sub = {v:Var(f"{v.name}_{Clause.next_index}") for v in self.logical_variables}
            Clause.next_index += 1
            return Clause(apply(self.head,sub),apply(self.body,sub))
        else:
           return self
        
def log_vars(exp, vs):
    """the union the logical variables in exp and the set vs"""
    if isinstance(exp,Var):
        return {exp}|vs
    elif isinstance(exp,Atom):
        return log_vars(exp.name, log_vars(exp.args, vs))
    elif isinstance(exp,(list,tuple)):
        for e in exp:
            vs = log_vars(e, vs)
    return vs

unifdisp = Var(None) # for display

def unify(t1,t2):
    e = [(t1,t2)]
    s = {}  # empty dictionary
    while e:
        (a,b) = e.pop()
        unifdisp.display(2,f"unifying{(a,b)}, e={e},s={s}")
        if a != b:
            if isinstance(a,Var):
                e = apply(e,{a:b}) 
                s = apply(s,{a:b})
                s[a]=b
            elif isinstance(b,Var):
                e = apply(e,{b:a}) 
                s = apply(s,{b:a})
                s[b]=a
            elif isinstance(a,Atom) and isinstance(b,Atom) and a.name==b.name and len(a.args)==len(b.args):
                e += zip(a.args,b.args)
            elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b ):
                e += zip(a,b)
            else:
                return False
    return s

def apply(e,sub):
    """e is an expression
    sub is a {var:val} dictionary
    returns e with all occurrence of var replaces with val"""
    if isinstance(e,Var) and e in sub:
        return sub[e]
    if isinstance(e,Atom):
        return Atom(e.name, apply(e.args,sub))
    if isinstance(e,list):
        return [apply(a,sub) for a in e]
    if isinstance(e,tuple):
        return tuple(apply(a,sub) for a in e)
    if isinstance(e,dict):
        return {k:apply(v,sub) for (k,v) in e.items()}
    else:
        return e

### Test cases:
# unifdisp.max_display_level = 2   # show trace
e1 = Atom('p',[Var('X'),Var('Y'),Var('Y')])
e2 = Atom('p',['a',Var('Z'),'b'])
# apply(e1,{Var('Y'):'b'})
# unify(e1,e2)
e3 = Atom('p',['a',Var('Y'),Var('Y')])
e4 = Atom('p',[Var('Z'),Var('Z'),'b'])
# unify(e3,e4)

class KB(logicProblem.KB):
    """A first-order knowledge base. 
      only the indexing is changed to index on name of the head."""
        
    def add_clause(self, c):
        """Add clause c to clause dictionary"""
        if c.head.name in self.atom_to_clauses:
            self.atom_to_clauses[c.head.name].append(c)
        else:
            self.atom_to_clauses[c.head.name] = [c]

    def ask(self, query):
        """self is the current KB
        query is a list of atoms to be proved
        generates {variable:value} dictionary"""

        qvars = list(log_vars(query, set()))
        for ans in self.prove(qvars, query):
            yield {x:v for (x,v) in zip(qvars,ans)}

    def ask_all(self, query):
        """returns a list of all answers to the query given kb"""
        return list(self.ask(query))

    def ask_one(self, query):
        """returns an answer to the query given kb or None of there are no answers"""
        for ans in self.ask(query):
            return ans

    def prove(self, ans, ans_body, indent=""):
        """enumerates the proofs for ans_body
        ans_body is a list of atoms to be proved
        ans is the list of values of the query variables
        """
        self.display(2,indent,f"(yes({ans}) <-"," & ".join(str(a) for a in ans_body))
        if ans_body==[]:
            yield ans
        else:
            selected, remaining = self.select_atom(ans_body)
            if self.built_in(selected):
                yield from self.eval_built_in(ans, selected, remaining, indent)
            else:
                for chosen_clause in self.atom_to_clauses[selected.name]:
                    clause = chosen_clause.rename()  # rename variables
                    sub = unify(selected, clause.head)
                    if sub is not False:
                        self.display(3,indent,"KB.prove: selected=", selected, "clause=",clause,"sub=",sub) 
                        resans = apply(ans,sub)
                        new_ans_body = apply(clause.body+remaining, sub)
                        yield from self.prove(resans, new_ans_body, indent+"    ")

    def select_atom(self,lst):
        """given list of atoms, return (selected atom, remaining atoms)
        """
        return lst[0],lst[1:]

    def built_in(self,atom):
        return atom.name in ['lt','triple']

    def eval_built_in(self,ans, selected, remaining, indent):
        if selected.name == 'lt':  # less than
            [a1,a2] = selected.args
            if a1 < a2:
                yield from self.prove(ans, remaining, indent+"    ")
        if selected.name == 'triple':    # use triple store (AIFCA Ch 16)
            yield from self.eval_triple(ans, selected, remaining, indent)

A = Var('A')
F = Var('F')
L  =Var('L')
W = Var('W')
X = Var('X')
Y = Var('Y')
Z = Var('Z')
def cons(h,t): return Atom('cons',[h,t])
def append(a,b,c): return Atom('append',[a,b,c])

app_KB = KB([
    Clause(append('nil',W,W)),
    Clause(append(cons(A,X), Y,cons(A,Z)),
                [append(X,Y,Z)])
    ])


lst = cons('l',cons('i',cons('s',cons('t','nil'))))
# app_KB.max_display_level = 2  #show derivation
#app_KB.ask_all([append(F,cons(A,'nil'), lst)])
# Think about the expected answer before trying:
#app_KB.ask_all([append(X, Y, lst)])
#app_KB.ask_all([append(lst, lst, L), append(X, cons('s',Y), L)])

