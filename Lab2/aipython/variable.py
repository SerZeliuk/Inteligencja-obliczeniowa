# variable.py - Representations of a variable in CSPs and probabilistic models
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random

class Variable(object):
    """A random variable.
    name (string) - name of the variable
    domain (list) - a list of the values for the variable.
    an (x,y) position for displaying
    """

    def __init__(self, name, domain, position=None):
        """Variable
        name a string
        domain a list of printable values
        position of form (x,y) where 0 <= x <= 1, 0 <= y <= 1
        """
        self.name = name   # string
        self.domain = domain # list of values
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain) 

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name  # f"Variable({self.name})"

