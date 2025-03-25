# logicNegation.py - Propositional negation-as-failure
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from logicProblem import KB, Clause, Askable, yes

class Not(object):
    def __init__(self, atom):
        self.theatom = atom

    def atom(self):
        return self.theatom

    def __repr__(self):
        return f"Not({self.theatom})"

def prove_naf(kb, ans_body, indent=""):
    """ prove with negation-as-failure and askables
    returns True if kb |- ans_body
    ans_body is a list of atoms to be proved
    """
    kb.display(2,indent,'yes <-',' & '.join(str(e) for e in ans_body))
    if ans_body:
        selected = ans_body[0]   # select first atom from ans_body
        if isinstance(selected, Not):
            kb.display(2,indent,f"proving {selected.atom()}")
            if prove_naf(kb, [selected.atom()], indent):
                kb.display(2,indent,f"{selected.atom()} succeeded so Not({selected.atom()}) fails")
                return False
            else:
                kb.display(2,indent,f"{selected.atom()} fails so Not({selected.atom()}) succeeds")
                return prove_naf(kb, ans_body[1:],indent+"    ")
        if selected in kb.askables:
            return (yes(input("Is "+selected+" true? "))
                    and  prove_naf(kb,ans_body[1:],indent+"    "))
        else:
            return any(prove_naf(kb,cl.body+ans_body[1:],indent+"    ")
                       for cl in kb.clauses_for_atom(selected))
    else:
        return True    # empty body is true

triv_KB_naf = KB([
    Clause('i_am', ['i_think']),
    Clause('i_think'),
    Clause('i_smell', ['i_am', Not('dead')]),
    Clause('i_bad', ['i_am', Not('i_think')])
    ])

triv_KB_naf.max_display_level = 4
def test():
    a1 = prove_naf(triv_KB_naf,['i_smell'])
    assert a1, f"triv_KB_naf failed to prove i_smell; gave {a1}"
    a2 = prove_naf(triv_KB_naf,['i_bad'])
    assert not a2, f"triv_KB_naf wrongly proved i_bad; gave {a2}"
    print("Passed unit tests")
if __name__ == "__main__":
    test()   

beach_KB = KB([
   Clause('away_from_beach', [Not('on_beach')]),
   Clause('beach_access', ['on_beach', Not('ab_beach_access')]),
   Clause('swim_at_beach', ['beach_access', Not('ab_swim_at_beach')]),
   Clause('ab_swim_at_beach', ['enclosed_bay', 'big_city', Not('ab_no_swimming_near_city')]),
   Clause('ab_no_swimming_near_city', ['in_BC', Not('ab_BC_beaches')])
    ])

# prove_naf(beach_KB, ['away_from_beach'])
# prove_naf(beach_KB, ['beach_access'])
# beach_KB.add_clause(Clause('on_beach',[]))
# prove_naf(beach_KB, ['away_from_beach'])
# prove_naf(beach_KB, ['swim_at_beach'])
# beach_KB.add_clause(Clause('enclosed_bay',[]))
# prove_naf(beach_KB, ['swim_at_beach'])
# beach_KB.add_clause(Clause('big_city',[]))
# prove_naf(beach_KB, ['swim_at_beach'])
# beach_KB.add_clause(Clause('in_BC',[]))
# prove_naf(beach_KB, ['swim_at_beach'])

