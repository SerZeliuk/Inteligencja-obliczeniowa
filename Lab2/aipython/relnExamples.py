# relnExamples.py - Relational Knowledge Base Example
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from logicRelation import Var, Atom, Clause, KB

simp_KB = KB([
    Clause(Atom('in',['kim','r123'])),
    Clause(Atom('part_of',['r123','cs_building'])),
    Clause(Atom('in',[Var('X'),Var('Y')]),
                   [Atom('part_of',[Var('Z'),Var('Y')]),
                    Atom('in',[Var('X'),Var('Z')])])
    ])

# define abbreviations to make the clauses more readable:
def lit(x): return Atom('lit',[x])
def light(x): return Atom('light',[x])
def ok(x): return Atom('ok',[x])
def live(x): return Atom('live',[x])
def connected_to(x,y): return Atom('connected_to',[x,y])
def up(x): return Atom('up',[x])
def down(x): return Atom('down',[x])

L = Var('L')
W = Var('W')
W1 = Var('W1')

elect_KB = KB([
    # lit(L) is true if light L is lit.
    Clause(lit(L),
               [light(L),
                ok(L),
                live(L)]),

    # live(W) is true if W is live (i.e., current will flow through it)
    Clause(live(W),
               [connected_to(W,W1),
                live(W1)]),

    Clause(live('outside')),

    # light(L) is true if L is a light
    Clause(light('l1')),
    Clause(light('l2')),

    # connected_to(W0,W1) is true if W0 is connected to W1 such that
    # current will flow from W1 to W0.

    Clause(connected_to('l1','w0')),
    Clause(connected_to('w0','w1'),
               [ up('s2'), ok('s2')]),
    Clause(connected_to('w0','w2'),
               [ down('s2'), ok('s2')]),
    Clause(connected_to('w1','w3'),
               [ up('s1'), ok('s1')]),
    Clause(connected_to('w2','w3'),
               [ down('s1'), ok('s1')]),
    Clause(connected_to('l2','w4')),
    Clause(connected_to('w4','w3'),
               [ up('s3'), ok('s3')]),
    Clause(connected_to('p1','w3')),
    Clause(connected_to('w3','w5'),
               [ ok('cb1')]),
    Clause(connected_to('p2','w6')),
    Clause(connected_to('w6','w5'),
               [ ok('cb2')]),
    Clause(connected_to('w5','outside'),
               [ ok('outside_connection')]),

    # up(S) is true if switch S is up
    # down(S) is true if switch S is down
    Clause(down('s1')),
    Clause(up('s2')),
    Clause(up('s3')),

    # ok(L) is true if K is working. Everything is ok:
    Clause(ok(L)),
    ])

# Example Queries:
# simp_KB.max_display_level = 2   # show trace
# ask_all(simp_KB, [Atom('in',[Var('A'),Var('B')])])

A = Var('A')
B = Var('B')

def test_ask_all(kb=simp_KB,
                     query=[Atom('in',[A,B])],
                     res=[{ A:'kim',B:'r123'}, {A:'kim',B: 'cs_building'}]):
    ans= kb.ask_all(query)
    assert ans == res, f"ask_all({query}) gave answer {ans}"
    print("ask_all: Passed unit test")

if __name__ == "__main__":
    test_ask_all()

# elect_KB.max_display_level = 2   # show trace
# elect_KB.ask_all([light('l1')])
# elect_KB.ask_all([light('l6')])
# elect_KB.ask_all([up(Var('X'))])
# elect_KB.ask_all([connected_to('w0',W)])
# elect_KB.ask_all([connected_to('w1',W)])
# elect_KB.ask_all([connected_to(W,'w3')])
# elect_KB.ask_all([connected_to(W1,W)])
# elect_KB.ask_all([live('w6')])
# elect_KB.ask_all([live('p1')])
# elect_KB.ask_all([Atom('lit',[L])])
# elect_KB.ask_all([Atom('lit',['l2']), live('p1')])
# elect_KB.ask_all([live(L)])
    
