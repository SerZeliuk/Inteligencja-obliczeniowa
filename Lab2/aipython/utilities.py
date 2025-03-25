# utilities.py - AIPython useful utilities
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
import math

def argmaxall(gen):
    """gen is a generator of (element,value) pairs, where value is a real.
    argmaxall returns a list of all of the elements with maximal value.
    """
    maxv = -math.inf       # negative infinity
    maxvals = []      # list of maximal elements
    for (e,v) in gen:
        if v > maxv:
            maxvals, maxv = [e], v
        elif v == maxv:
            maxvals.append(e)
    return maxvals

def argmaxe(gen):
    """gen is a generator of (element,value) pairs, where value is a real.
    argmaxe returns an element with maximal value.
    If there are multiple elements with the max value, one is returned at random.
    """
    return random.choice(argmaxall(gen))

def argmax(lst):
    """returns maximum index in a list"""
    return argmaxe(enumerate(lst))
# Try:
# argmax([1,6,3,77,3,55,23])

def argmaxd(dct):
   """returns the arg max of a dictionary dct"""
   return argmaxe(dct.items())
# Try:
# arxmaxd({2:5,5:9,7:7})
def flip(prob):
    """return true with probability prob"""
    return random.random() < prob

def select_from_dist(item_prob_dist):
    """ returns a value from a distribution.
    item_prob_dist is an item:probability dictionary, where the
        probabilities sum to 1.
    returns an item chosen in proportion to its probability
    """
    ranreal = random.random()
    for (it,prob) in item_prob_dist.items():
        if ranreal < prob:
            return it
        else:
            ranreal -= prob
    raise RuntimeError(f"{item_prob_dist} is not a probability distribution")

def test():
    """Test part of utilities"""
    assert argmax([1,6,55,3,55,23]) in [2,4]
    print("Passed unit test in utilities")
    print("run test_aipython() to test (almost) everything")

if __name__ == "__main__":
    test()

def test_aipython():
    import pythonDemo, display
    # Agents: currently no tests
    import agents, agentBuying, agentEnv, agentMiddle, agentTop, agentFollowTarget
    # Search:
    print("***** testing Search *****")
    import searchGeneric, searchBranchAndBound, searchExample, searchTest
    searchGeneric.test(searchGeneric.AStarSearcher)
    searchBranchAndBound.test(searchBranchAndBound.DF_branch_and_bound)
    searchTest.run(searchExample.problem1,"Problem 1")
    import searchGUI, searchMPP, searchGrid
    # CSP
    print("\n***** testing CSP *****")
    import cspExamples, cspDFS, cspSearch, cspConsistency, cspSLS
    cspExamples.test_csp(cspDFS.dfs_solve1)
    cspExamples.test_csp(cspSearch.solver_from_searcher)
    cspExamples.test_csp(cspConsistency.ac_solver)
    cspExamples.test_csp(cspConsistency.ac_search_solver)
    cspExamples.test_csp(cspSLS.sls_solver) 
    cspExamples.test_csp(cspSLS.any_conflict_solver)
    import cspConsistencyGUI, cspSoft
    # Propositions
    print("\n***** testing Propositional Logic *****")
    import logicBottomUp, logicTopDown, logicExplain, logicAssumables, logicNegation
    logicBottomUp.test()
    logicTopDown.test()
    logicExplain.test()
    logicNegation.test()
    # Planning
    print("\n***** testing Planning *****")
    import stripsHeuristic
    stripsHeuristic.test_forward_heuristic()
    stripsHeuristic.test_regression_heuristic()
    import stripsCSPPlanner, stripsPOP
    # Learning
    print("\n***** testing Learning *****")
    import learnProblem, learnNoInputs, learnDT, learnLinear
    learnNoInputs.test_no_inputs(training_sizes=[4])
    data = learnProblem.Data_from_file('data/carbool.csv', target_index=-1, seed=123)
    learnDT.testDT(data, print_tree=False)
    learnLinear.test()
    import learnCrossValidation, learnBoosting
    # Deep Learning: currently no tests
    import learnNN
    # Uncertainty
    print("\n***** testing Uncertainty *****")
    import probGraphicalModels, probRC, probVE, probStochSim
    probGraphicalModels.InferenceMethod.testIM(probRC.ProbSearch)
    probGraphicalModels.InferenceMethod.testIM(probRC.ProbRC)
    probGraphicalModels.InferenceMethod.testIM(probVE.VE)
    probGraphicalModels.InferenceMethod.testIM(probStochSim.RejectionSampling, threshold=0.1)
    probGraphicalModels.InferenceMethod.testIM(probStochSim.LikelihoodWeighting, threshold=0.1)
    probGraphicalModels.InferenceMethod.testIM(probStochSim.ParticleFiltering, threshold=0.1)
    probGraphicalModels.InferenceMethod.testIM(probStochSim.GibbsSampling, threshold=0.1)
    import probHMM, probLocalization, probDBN
    # Learning under uncertainty: currently no tests
    import learnBayesian, learnKMeans, learnEM
    # Causality: currently no tests
    import probDo, probCounterfactual
    # Planning under uncertainty
    print("\n***** testing Planning under Uncertainty *****")
    import decnNetworks
    decnNetworks.test(decnNetworks.fire_dn)
    import mdpExamples
    mdpExamples.test_MDP(mdpExamples.partyMDP)
    import mdpGUI
    # Reinforcement Learning:
    print("\n***** testing Reinforcement Learning *****")
    import rlQLearner
    rlQLearner.test_RL(rlQLearner.Q_learner, alpha_fun=lambda k:10/(9+k))
    import rlQExperienceReplay
    rlQLearner.test_RL(rlQExperienceReplay.Q_ER_learner, alpha_fun=lambda k:10/(9+k))
    import rlStochasticPolicy
    rlQLearner.test_RL(rlStochasticPolicy.StochasticPIAgent, alpha_fun=lambda k:10/(9+k))
    import rlModelLearner
    rlQLearner.test_RL(rlModelLearner.Model_based_reinforcement_learner)
    import rlFeatures
    rlQLearner.test_RL(rlFeatures.SARSA_LFA_learner, es_kwargs={'epsilon':1}, eps=4)
    import rlQExperienceReplay, rlModelLearner, rlFeatures, rlGUI
    # Multiagent systems: currently no tests
    import rlStochasticPolicy, rlGameFeature
    # Individuals and Relations
    print("\n***** testing Datalog and Logic Programming *****")
    import relnExamples
    relnExamples.test_ask_all()
    # Knowledge Graphs and Ontologies
    print("\n***** testing Knowledge Graphs and Ontologies *****")
    import knowledgeGraph, knowledgeReasoning
    knowledgeGraph.test_kg()
    # Relational Learning: currently no tests
    import relnCollFilt, relnProbModels
    print("\n***** End of Testing*****")
    
