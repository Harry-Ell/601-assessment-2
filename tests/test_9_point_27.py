"""
Python test file for exercise 9.27
"""
import numpy as np
from numpy.testing import assert_array_equal
from value_iteration.MDP import GenericMDP #type: ignore

# for 9.27, we simply write out what we know the tarket policy should be 
def test_ex_9_27():
    '''
    test for the code to see if it can return the same output for this particular case
    '''
    states = ['Healthy', 'Sick']
    actions = ['Relax', 'Party']
    probabilities = {'Healthy': {0: {'Healthy': 0.95, 'Sick': 0.05},
                                 1: {'Healthy': 0.7, 'Sick': 0.300}},
                    'Sick':     {0: {'Healthy': 0.5, 'Sick': 0.5}, 
                                 1: {'Healthy': 0.1, 'Sick': 0.9}}}
    
    rewards = {'Healthy': {0: {'Healthy': 10.0, 'Sick': 10.0},
                           1: {'Healthy': 7.0, 'Sick': 7.0}},
               'Sick':    {0: {'Healthy': 0.0, 'Sick': 0.0}, 
                           1: {'Healthy': 2.0, 'Sick': 2.0}}}
    
    target_policy = {'Healthy': (np.array([0]),), 'Sick': (np.array([0]),)}
    target_values = {'Healthy': np.float64(46.87499618960981), 'Sick': np.float64(31.24999618960981)}

    mdp_solver = GenericMDP(states, actions, probabilities, rewards, 0.8, 200)
    policy, values = mdp_solver()
    assert policy.keys() == target_policy.keys()
    for key in policy:
        assert_array_equal(policy[key][0], policy[key][0])

    assert values.keys() == target_values.keys()
    for key in policy:
        assert round(values[key], 3) == round(target_values[key], 3)
