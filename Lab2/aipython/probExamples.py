# probExamples.py - Example belief networks
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from variable import Variable
from probFactors import CPD, Prob, LogisticRegression, NoisyOR, ConstantCPD
from probGraphicalModels import BeliefNetwork

# Belief network report-of-leaving example (Example 9.13 shown in Figure 9.3) of
# Poole and Mackworth, Artificial Intelligence, 2023 http://artint.info
boolean = [False, True]

Alarm =   Variable("Alarm",   boolean,  position=(0.366,0.5))
Fire =    Variable("Fire",    boolean,  position=(0.633,0.75))
Leaving = Variable("Leaving", boolean,  position=(0.366,0.25))
Report =  Variable("Report",  boolean,  position=(0.366,0.0))
Smoke =   Variable("Smoke",   boolean,  position=(0.9,0.5))
Tamper =  Variable("Tamper",  boolean,  position=(0.1,0.75))

f_ta = Prob(Tamper,[],[0.98,0.02])
f_fi = Prob(Fire,[],[0.99,0.01])
f_sm = Prob(Smoke,[Fire],[[0.99,0.01],[0.1,0.9]])
f_al = Prob(Alarm,[Fire,Tamper],[[[0.9999, 0.0001], [0.15, 0.85]], [[0.01, 0.99], [0.5, 0.5]]])
f_lv = Prob(Leaving,[Alarm],[[0.999, 0.001], [0.12, 0.88]])
f_re = Prob(Report,[Leaving],[[0.99, 0.01], [0.25, 0.75]])

bn_report = BeliefNetwork("Report-of-leaving", {Tamper,Fire,Smoke,Alarm,Leaving,Report},
                              {f_ta,f_fi,f_sm,f_al,f_lv,f_re})

# Belief network simple-diagnostic example (Exercise 9.3 shown in Figure 9.39) of
# Poole and Mackworth, Artificial Intelligence, 2023 http://artint.info

Influenza =   Variable("Influenza",   boolean,  position=(0.4,0.8))
Smokes =      Variable("Smokes",   boolean,  position=(0.8,0.8))
SoreThroat =  Variable("Sore Throat",   boolean,  position=(0.2,0.5))
HasFever =       Variable("Fever",   boolean,  position=(0.4,0.5))
Bronchitis =  Variable("Bronchitis",   boolean,  position=(0.6,0.5))
Coughing =    Variable("Coughing",   boolean,  position=(0.4,0.2))
Wheezing =    Variable("Wheezing",   boolean,  position=(0.8,0.2))

p_infl =    Prob(Influenza,[],[0.95,0.05])
p_smokes =  Prob(Smokes,[],[0.8,0.2])
p_sth =     Prob(SoreThroat,[Influenza],[[0.999,0.001],[0.7,0.3]])
p_fever =   Prob(HasFever,[Influenza],[[0.99,0.05],[0.9,0.1]])
p_bronc = Prob(Bronchitis,[Influenza,Smokes],[[[0.9999, 0.0001], [0.3, 0.7]], [[0.1, 0.9], [0.01, 0.99]]])
p_cough =   Prob(Coughing,[Bronchitis],[[0.93,0.07],[0.2,0.8]])
p_wheeze =   Prob(Wheezing,[Bronchitis],[[0.999,0.001],[0.4,0.6]])

simple_diagnosis = BeliefNetwork("Simple Diagnosis",
                    {Influenza, Smokes, SoreThroat, HasFever, Bronchitis, Coughing, Wheezing},
                    {p_infl, p_smokes, p_sth, p_fever, p_bronc, p_cough, p_wheeze})

Season = Variable("Season", ["dry_season","wet_season"],  position=(0.5,0.9))
Sprinkler = Variable("Sprinkler", ["on","off"],  position=(0.9,0.6))
Rained = Variable("Rained", boolean,  position=(0.1,0.6))
Grass_wet = Variable("Grass wet", boolean,  position=(0.5,0.3))
Grass_shiny = Variable("Grass shiny", boolean,  position=(0.1,0))
Shoes_wet = Variable("Shoes wet", boolean,  position=(0.9,0))

f_season = Prob(Season,[],{'dry_season':0.5, 'wet_season':0.5})
f_sprinkler = Prob(Sprinkler,[Season],{'dry_season':{'on':0.4,'off':0.6},
                                       'wet_season':{'on':0.01,'off':0.99}})
f_rained = Prob(Rained,[Season],{'dry_season':[0.9,0.1], 'wet_season': [0.2,0.8]})
f_wet = Prob(Grass_wet,[Sprinkler,Rained], {'on': [[0.1,0.9],[0.01,0.99]],
                                            'off':[[0.99,0.01],[0.3,0.7]]})
f_shiny = Prob(Grass_shiny, [Grass_wet], [[0.95,0.05], [0.3,0.7]])
f_shoes = Prob(Shoes_wet, [Grass_wet], [[0.98,0.02], [0.35,0.65]])

bn_sprinkler = BeliefNetwork("Pearl's Sprinkler Example",
                         {Season, Sprinkler, Rained, Grass_wet, Grass_shiny, Shoes_wet},
                         {f_season, f_sprinkler, f_rained, f_wet, f_shiny, f_shoes})

#### Bipartite Diagnostic Network ###
Cough = Variable("Cough", boolean, (0.1,0.1))
Fever = Variable("Fever", boolean, (0.5,0.1))
Sneeze = Variable("Sneeze", boolean, (0.9,0.1))
Cold = Variable("Cold",boolean, (0.1,0.9))
Flu = Variable("Flu",boolean, (0.5,0.9))
Covid = Variable("Covid",boolean, (0.9,0.9))

p_cold_no = Prob(Cold,[],[0.9,0.1])
p_flu_no = Prob(Flu,[],[0.95,0.05])
p_covid_no = Prob(Covid,[],[0.99,0.01])

p_cough_no = NoisyOR(Cough,   [Cold,Flu,Covid], [0.1,  0.3,  0.2,  0.7])
p_fever_no = NoisyOR(Fever,   [     Flu,Covid], [0.01,       0.6,  0.7])
p_sneeze_no = NoisyOR(Sneeze, [Cold,Flu      ], [0.05,  0.5,  0.2    ])

bn_no1 = BeliefNetwork("Bipartite Diagnostic Network (noisy-or)",
                         {Cough, Fever, Sneeze, Cold, Flu, Covid},
                          {p_cold_no, p_flu_no, p_covid_no, p_cough_no, p_fever_no, p_sneeze_no})  

# to see the conditional probability of Noisy-or do:
# print(p_cough_no.to_table())

# example from box "Noisy-or compared to logistic regression"
# X = Variable("X",boolean)
# w0 = 0.01
# print(NoisyOR(X,[A,B,C,D],[w0, 1-(1-0.05)/(1-w0), 1-(1-0.1)/(1-w0), 1-(1-0.2)/(1-w0), 1-(1-0.2)/(1-w0), ]).to_table(given={X:True}))


p_cold_lr = Prob(Cold,[],[0.9,0.1])
p_flu_lr = Prob(Flu,[],[0.95,0.05])
p_covid_lr = Prob(Covid,[],[0.99,0.01])

p_cough_lr =  LogisticRegression(Cough,  [Cold,Flu,Covid], [-2.2,  1.67,  1.26,  3.19])
p_fever_lr =  LogisticRegression(Fever,  [     Flu,Covid], [-4.6,         5.02,  5.46])
p_sneeze_lr = LogisticRegression(Sneeze, [Cold,Flu      ], [-2.94, 3.04,  1.79    ])

bn_lr1 = BeliefNetwork("Bipartite Diagnostic Network -  logistic regression",
                         {Cough, Fever, Sneeze, Cold, Flu, Covid},
                          {p_cold_lr, p_flu_lr, p_covid_lr, p_cough_lr, p_fever_lr, p_sneeze_lr})  

# to see the conditional probability of Noisy-or do:
#print(p_cough_lr.to_table())

# example from box "Noisy-or compared to logistic regression"
# from learnLinear import sigmoid, logit
# w0=logit(0.01)
# X = Variable("X",boolean)
# print(LogisticRegression(X,[A,B,C,D],[w0, logit(0.05)-w0, logit(0.1)-w0, logit(0.2)-w0, logit(0.2)-w0]).to_table(given={X:True}))
# try to predict what would happen (and then test) if we had
# w0=logit(0.01)

