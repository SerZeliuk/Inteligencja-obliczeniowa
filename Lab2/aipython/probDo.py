# probDo.py - Probabilistic inference with the do operator
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from probGraphicalModels import InferenceMethod, BeliefNetwork
from probFactors import CPD, ConstantCPD

def intervene(bn, do={}):
    assert isinstance(bn, BeliefNetwork), f"Do only applies to  belief networks ({bn.title})"
    if do=={}:
        return bn
    else:
        newfacs = ({f for (ch,f) in bn.var2cpt.items() if ch not in do} |
                       {ConstantCPD(v,c) for (v,c) in do.items()})
        return BeliefNetwork(f"{bn.title}(do={do})", bn.variables, newfacs)

def queryDo(self, qvar, obs={}, do={}):
    """Extends query method to also allow for interventions.
    """
    oldBN, self.gm = self.gm, intervene(self.gm, do)
    result = self.query(qvar, obs)
    self.gm = oldBN  # restore original
    return result

# make queryDo available for all inference methods
InferenceMethod.queryDo = queryDo
    
from probRC import ProbRC

from probExamples import bn_sprinkler, Season, Sprinkler, Rained, Grass_wet, Grass_shiny, Shoes_wet
bn_sprinklerv = ProbRC(bn_sprinkler)
## bn_sprinklerv.queryDo(Shoes_wet)
## bn_sprinklerv.queryDo(Shoes_wet,obs={Sprinkler:"on"})
## bn_sprinklerv.queryDo(Shoes_wet,do={Sprinkler:"on"})
## bn_sprinklerv.queryDo(Season, obs={Sprinkler:"on"})
## bn_sprinklerv.queryDo(Season, do={Sprinkler:"on"})

### Showing posterior distributions:
# bn_sprinklerv.show_post({})
# bn_sprinklerv.show_post({Sprinkler:"on"})
# spon = intervene(bn_sprinkler, do={Sprinkler:"on"})
# ProbRC(spon).show_post({})

from variable import Variable
from probFactors import Prob
from probGraphicalModels import BeliefNetwork
boolean = [False, True]

Drug_Prone = Variable("Drug_Prone", boolean, position=(0.1,0.5)) # (0.5,0.9))
Side_Effects = Variable("Side_Effects", boolean, position=(0.1,0.5)) # (0.5,0.1))
Takes_Marijuana = Variable("\nTakes_Marijuana\n", boolean, position=(0.1,0.5))
Takes_Hard_Drugs = Variable("Takes_Hard_Drugs", boolean, position=(0.9,0.5))

p_dp = Prob(Drug_Prone, [], [0.8, 0.2])
p_be = Prob(Side_Effects, [Takes_Marijuana], [[1, 0], [0.4, 0.6]])
p_tm = Prob(Takes_Marijuana, [Drug_Prone], [[0.98, 0.02], [0.2, 0.8]])
p_thd = Prob(Takes_Hard_Drugs, [Side_Effects, Drug_Prone],
                 # Drug_Prone=False    Drug_Prone=True
                 [[[0.999, 0.001],     [0.6, 0.4]], # Side_Effects=False
                  [[0.99999, 0.00001], [0.995, 0.005]]])  # Side_Effects=True

drugs = BeliefNetwork("Gateway Drug?",
                    [Drug_Prone,Side_Effects, Takes_Marijuana, Takes_Hard_Drugs],
                    [p_tm, p_dp, p_be, p_thd])
                    
drugsq = ProbRC(drugs)
# drugsq.queryDo(Takes_Hard_Drugs)
# drugsq.queryDo(Takes_Hard_Drugs, obs = {Takes_Marijuana: True})
# drugsq.queryDo(Takes_Hard_Drugs, obs = {Takes_Marijuana: False})
# drugsq.queryDo(Takes_Hard_Drugs, do = {Takes_Marijuana: True})
# drugsq.queryDo(Takes_Hard_Drugs, do = {Takes_Marijuana: False})

# ProbRC(drugs).show_post({})
# ProbRC(drugs).show_post({Takes_Marijuana: True})
# ProbRC(drugs).show_post({Takes_Marijuana: False})
# ProbRC(intervene(drugs, do={Takes_Marijuana: True})).show_post({})
# ProbRC(intervene(drugs, do={Takes_Marijuana: False})).show_post({})
# Why was that? Try the following then repeat:
# Drug_Prone.position=(0.5,0.9); Side_Effects.position=(0.5,0.1)

