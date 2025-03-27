"""
Value iteration implementation in python. 

Specifically optimised for the case of gridworld style problems, 
comes with some slightly unneccessary overhead in the case of simpler problems. 

Interactable with via a command line interface. 

More full exposition given on repository homepage: 

Or on Pypi page:


"""

from .Gridworld_Constructor import Gridworld_Constructor
from .Markov_Decision_Process import Value_Iteration
from .MDP import GenericMDP

__version__ = '1.0.1'
__author__ = 'Harry Ellingham'