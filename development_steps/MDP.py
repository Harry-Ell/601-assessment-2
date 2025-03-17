import numpy as np 

class GenericMDP:

    def __init__(self, states, actions, probabilities, rewards, discount_rate, max_iterations):
        self.states = states
        self.actions = actions
        self.probabilities = probabilities
        self.discount_rate = discount_rate
        self.max_iter = max_iterations
        self.rewards = rewards
        # self.problem_type = problem_type

        # # in the case of a grid world
        # self.border_penalty = kwargs.get('border_penalty', -1)
        # self.other = kwargs.get('other', -1)
        # self.other_other = kwargs.get('other', -1)


    def _bellmans_eq(self, state, values_dict:dict):
        '''
        This will be a generic bellmans which takes in the probabilities, rewards and states
        
        intentionally sparse for the time being to avoid wasted computation
        '''

        action_space = len(self.actions)
        V_k = np.zeros(action_space) 

        for action in range(action_space):
            V_k[action] = self.rewards[state][action] + self.discount_rate * \
                          sum(self.probabilities[state][action][s_prime] * values_dict.get(s_prime, 0) \
                          for s_prime in self.probabilities[state][action])
            
        return max(V_k) 

    def _value_iteration(self):
        '''value iteration function which we have to 
        
        soon, this will need termination conditions'''

        V_k = {}   
        k = 0
        while k < self.max_iter:
            V_k_minus_1 = V_k.copy()
            for state in self.states:
                V_k[state] = self._bellmans_eq(state = state, values_dict = V_k_minus_1)
                
            k += 1
        return V_k
    
    def _extract_policy(self):
        '''
        Description: 
            We will only do this once as to reduce computational load on the solver
        Inputs: 
            Value function
        Returns: 
            Policy
        '''


    def __call__(self, plot_grid, plot_policy):
        '''call funciton which we are meant to be able to use to '''



class MDPConstructor:
    '''here we can contain all of the helper functions which will allow us to make the transition matricies we need'''
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.border_penalty = kwargs.get('border_penalty', -1)
        self.reward_states = kwargs.get('reward_states', None)


    def _grid_world(self):
        '''class to populate everything in the case of a grid world example'''
        probabilities, rewards = None, None
        return probabilities, rewards

    def _misc(self):
        '''in the case of a non grid world problem, we still need to make it so that we have probabilities specified in 
        some reasonable fashion'''
        pass

    def __call__(self, problem_type):
        '''the idea here is to funnel you down to the correct constructors'''

