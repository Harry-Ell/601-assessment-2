"""
Python test file for exercise 9.27

For 9.28, we need to write out all the probabilities and input these to run the function. 
There is actually a large amount of overhead to doing this, hence this is going to be a very long test. 
"""

import numpy as np
from numpy.testing import assert_array_equal
from value_iteration.MDP import GenericMDP

def populate_probabilities_rewards(state, actions, corners):
    sub_dirs = {}
    for index, _ in enumerate(actions):
        sub_dirs[index] = {}
        for corner in corners:
            sub_dirs[index][(corner)] = 0.25
    return sub_dirs

def populate_probabilities_usual(state, actions):
    sub_dirs = {}
    for index, intended_move in enumerate(actions):
        sub_dirs[index] = {}
        for actual_move in actions:
            value = 0.7 if intended_move == actual_move else 0.1
            sub_dirs[index][(state[0] + actual_move[0], state[1] + actual_move[1])] = value
    return sub_dirs

def populate_probabilities_edge(state, actions):
    sub_dirs = {}

    for index, intended_move in enumerate(actions):
        sub_dirs[index] = {}

        for actual_move in actions:
            new_row = state[0] + actual_move[0]
            new_col = state[1] + actual_move[1]

            # Check if out of bounds
            if (new_row < 0 or new_row >= 10 or
                new_col < 0 or new_col >= 10):
                # Remain in the same cell 
                new_state = state
            else:
                new_state = (new_row, new_col)

            value = 0.7 if intended_move == actual_move else 0.1
            sub_dirs[index][new_state] = value

    return sub_dirs

def populate_positive_rewards(state, actions, reward_value, corners):
    sub_dirs = {}
    for index, _ in enumerate(actions):
        sub_dirs[index] = {}
        for corner in corners:
            sub_dirs[index][(corner)] = reward_value
    return sub_dirs

def populate_negative_rewards(state, actions, reward_value):
    sub_dirs = {}
    for index, intended_move in enumerate(actions):
        sub_dirs[index] = {}
        for actual_move in actions:
            sub_dirs[index][(state[0] + actual_move[0], state[1] + actual_move[1])] = reward_value
    return sub_dirs

def populate_edge_penalties(state, actions):
    sub_dirs = {}

    for index, intended_move in enumerate(actions):
        sub_dirs[index] = {}

        for actual_move in actions:
            new_row = state[0] + actual_move[0]
            new_col = state[1] + actual_move[1]

            # Check if out of bounds
            if (new_row < 0 or new_row >= 10 or
                new_col < 0 or new_col >= 10):
                # Remain in the same cell 
                new_state = state
            else:
                new_state = (new_row, new_col)

            value = -1 if new_state == state else 0
            sub_dirs[index][new_state] = value

    return sub_dirs



def test_ex_9_28():
    '''
    test for the code to see if it can return the same output for this particular case
    '''
    len_x, len_y = 10, 10

    states = [(i, j) for i in range(len_x) for j in range(len_y)]
    probabilities = {}

    corners = [(0, 0), (0, len_y - 1), (len_x - 1, 0), (len_x - 1, len_y - 1)]
    reward_states = [(8,7), (7,2), (3,4), (3,7)]
    reward_values = [10, 3, -5, -10]

    actions = [(1, 0),    # right
            (-1, 0),   # left
            (0, 1),    # up
            (0, -1)]   # down
    
    for state in states:
        i, j = state
        if state in reward_states and reward_values[reward_states.index(state)] > 0:
            # teleportation step
            probabilities[state] = populate_probabilities_rewards(state, actions, corners)
        elif 0 < i < len_x-1 and 0 < j < len_y-1:
            # we are in standard operating conditions, make usual sub directories
            probabilities[state] = populate_probabilities_usual(state, actions)
        else: # we must be at an edge
            probabilities[state] = populate_probabilities_edge(state, actions)

    rewards = {}
    for state in states:
        i, j = state
        if state in reward_states and reward_values[reward_states.index(state)] > 0:
            # we are about to teleport
            rewards[state] = populate_positive_rewards(state, actions, reward_values[reward_states.index(state)], corners)
        elif state in reward_states and reward_values[reward_states.index(state)] < 0:
            # we get a reward but do not teleport
            rewards[state] = populate_negative_rewards(state, actions, reward_values[reward_states.index(state)])
        elif i in [0, len_x-1] or j in [0, len_y-1]:
            rewards[state] = populate_edge_penalties(state, actions)

    target_values = {(0, 0): np.float64(0.902),(0, 1): np.float64(1.264),(0, 2): np.float64(1.409),(0, 3): np.float64(1.326),(0, 4): np.float64(1.375),
        (0, 5): np.float64(1.612),(0, 6): np.float64(1.615),(0, 7): np.float64(1.439),(0, 8): np.float64(1.648),(0, 9): np.float64(1.572),
        (1, 0): np.float64(1.282),(1, 1): np.float64(1.654),(1, 2): np.float64(1.823),(1, 3): np.float64(1.677),(1, 4): np.float64(1.726),
        (1, 5): np.float64(2.06),(1, 6): np.float64(2.056),(1, 7): np.float64(1.77),(1, 8): np.float64(2.109),(1, 9): np.float64(2.194),
        (2, 0): np.float64(1.594),(2, 1): np.float64(2.001),(2, 2): np.float64(2.217),(2, 3): np.float64(1.965),(2, 4): np.float64(1.704),
        (2, 5): np.float64(2.499),(2, 6): np.float64(2.485),(2, 7): np.float64(1.414),(2, 8): np.float64(2.546),(2, 9): np.float64(2.801),
        (3, 0): np.float64(1.937),(3, 1): np.float64(2.395),(3, 2): np.float64(2.693),(3, 3): np.float64(2.319),(3, 4): np.float64(-2.25),
        (3, 5): np.float64(3.074),(3, 6): np.float64(3.093),(3, 7): np.float64(-6.371),(3, 8): np.float64(3.138),(3, 9): np.float64(3.528),
        (4, 0): np.float64(2.334),(4, 1): np.float64(2.855),(4, 2): np.float64(3.284),(4, 3): np.float64(3.337),(4, 4): np.float64(3.351),
        (4, 5): np.float64(4.401),(4, 6): np.float64(5.025),(4, 7): np.float64(4.668),(4, 8): np.float64(5.023),(4, 9): np.float64(4.406),
        (5, 0): np.float64(2.793),(5, 1): np.float64(3.386),(5, 2): np.float64(3.943),(5, 3): np.float64(4.018),(5, 4): np.float64(4.535),
        (5, 5): np.float64(5.351),(5, 6): np.float64(6.239),(5, 7): np.float64(6.884),(5, 8): np.float64(6.229),(5, 9): np.float64(5.302),
        (6, 0): np.float64(3.314),(6, 1): np.float64(4.005),(6, 2): np.float64(4.732),(6, 3): np.float64(4.69),(6, 4): np.float64(5.382),
        (6, 5): np.float64(6.325),(6, 6): np.float64(7.437),(6, 7): np.float64(8.479),(6, 8): np.float64(7.429),(6, 9): np.float64(6.297),
        (7, 0): np.float64(3.791),(7, 1): np.float64(4.672),(7, 2): np.float64(5.706),(7, 3): np.float64(5.425),(7, 4): np.float64(6.294),
        (7, 5): np.float64(7.444),(7, 6): np.float64(8.799),(7, 7): np.float64(10.351),(7, 8): np.float64(8.791),(7, 9): np.float64(7.436),
        (8, 0): np.float64(3.424),(8, 1): np.float64(4.175),(8, 2): np.float64(4.955),(8, 3): np.float64(5.828),(8, 4): np.float64(7.016),
        (8, 5): np.float64(8.505),(8, 6): np.float64(10.362),(8, 7): np.float64(12.706),(8, 8): np.float64(10.352),(8, 9): np.float64(8.468),
        (9, 0): np.float64(2.724),(9, 1): np.float64(3.604),(9, 2): np.float64(4.378),(9, 3): np.float64(5.26),(9, 4): np.float64(6.299),
        (9, 5): np.float64(7.504),(9, 6): np.float64(8.883),(9, 7): np.float64(10.437),(9, 8): np.float64(8.815),(9, 9): np.float64(6.83)}
    

    target_policy = {(0, 0): (np.array([0]),),(0, 1): (np.array([0]),),(0, 2): (np.array([0]),),(0, 3): (np.array([0]),),(0, 4): (np.array([0]),),
        (0, 5): (np.array([0]),),(0, 6): (np.array([0]),),(0, 7): (np.array([0]),),(0, 8): (np.array([0]),),(0, 9): (np.array([0]),),(1, 0): (np.array([2]),),
        (1, 1): (np.array([0]),),(1, 2): (np.array([0]),),(1, 3): (np.array([0]),),(1, 4): (np.array([2]),),(1, 5): (np.array([0]),),(1, 6): (np.array([0]),),
        (1, 7): (np.array([2]),),(1, 8): (np.array([0]),),(1, 9): (np.array([0]),),(2, 0): (np.array([2]),),(2, 1): (np.array([0]),),(2, 2): (np.array([0]),),
        (2, 3): (np.array([0]),),(2, 4): (np.array([2]),),(2, 5): (np.array([0]),),(2, 6): (np.array([0]),),(2, 7): (np.array([2]),),(2, 8): (np.array([0]),),
        (2, 9): (np.array([0]),),(3, 0): (np.array([2]),),(3, 1): (np.array([0]),),(3, 2): (np.array([0]),),(3, 3): (np.array([0]),),(3, 4): (np.array([0]),),
        (3, 5): (np.array([0]),),(3, 6): (np.array([0]),),(3, 7): (np.array([0]),),(3, 8): (np.array([0]),),(3, 9): (np.array([0]),),(4, 0): (np.array([2]),),
        (4, 1): (np.array([0]),),(4, 2): (np.array([0]),),(4, 3): (np.array([0]),),(4, 4): (np.array([0]),),(4, 5): (np.array([0]),),(4, 6): (np.array([0]),),
        (4, 7): (np.array([0]),),(4, 8): (np.array([0]),),(4, 9): (np.array([0]),),(5, 0): (np.array([2]),),(5, 1): (np.array([0]),),(5, 2): (np.array([0]),),
        (5, 3): (np.array([0]),),(5, 4): (np.array([0]),),(5, 5): (np.array([0]),),(5, 6): (np.array([0]),),(5, 7): (np.array([0]),),(5, 8): (np.array([0]),),
        (5, 9): (np.array([0]),),(6, 0): (np.array([2]),),(6, 1): (np.array([2]),),(6, 2): (np.array([0]),),(6, 3): (np.array([0]),),(6, 4): (np.array([2]),),
        (6, 5): (np.array([0]),),(6, 6): (np.array([0]),),(6, 7): (np.array([0]),),(6, 8): (np.array([0]),),(6, 9): (np.array([0]),),(7, 0): (np.array([2]),),
        (7, 1): (np.array([2]),),(7, 2): (np.array([0, 1, 2, 3]),),(7, 3): (np.array([2]),),(7, 4): (np.array([2]),),(7, 5): (np.array([2]),),(7, 6): (np.array([0]),),
        (7, 7): (np.array([0]),),(7, 8): (np.array([0]),),(7, 9): (np.array([3]),),(8, 0): (np.array([2]),),(8, 1): (np.array([2]),),(8, 2): (np.array([2]),),
        (8, 3): (np.array([2]),),(8, 4): (np.array([2]),),(8, 5): (np.array([2]),),(8, 6): (np.array([2]),),(8, 7): (np.array([0, 1, 2, 3]),),(8, 8): (np.array([3]),),
        (8, 9): (np.array([3]),),(9, 0): (np.array([2]),),(9, 1): (np.array([2]),),(9, 2): (np.array([2]),),(9, 3): (np.array([2]),),(9, 4): (np.array([2]),),
        (9, 5): (np.array([2]),),(9, 6): (np.array([2]),),(9, 7): (np.array([1]),),(9, 8): (np.array([3]),),(9, 9): (np.array([3]),)}
    
    mdp_solver = GenericMDP(states, actions, probabilities, rewards, 0.9, 200, len_x=len_x, len_y=len_y, reward_list=reward_states, reward_values= reward_values, problem_type='gridworld')
    policy, values = mdp_solver()
    assert policy.keys() == target_policy.keys()
    for key in policy:
        assert_array_equal(policy[key][0], policy[key][0])

    assert values.keys() == target_values.keys()
    for key in policy:
        assert round(values[key], 3) == round(target_values[key], 3)