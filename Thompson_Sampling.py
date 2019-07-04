import numpy as np
import scipy as sp
import scipy.stats as st
import Optimization as opt


class State:
    def __init__(self, K):
                
        self.Alpha = np.ones(K) # The alpha values of the arms
        self.Beta = np.ones(K)  # The beta values of the arms     
    
    def update(self, a, obs):
        """
        Update the Alpha and Beta arrays given action a and observation obs. 
        
        Input:
          a:    the action taken in round t, an integer in [K]
          obs:  the observation incurred in round t, 0 or 1
        """
        
        if obs == 1:
            self.Alpha[a] += 1
        else:
            self.Beta[a] += 1        
        return None
    
    def print(self):
        """
        Print the current value of the state variables."
        """
        
        print('Alpha', self.Alpha)
        print('Beta', self.Beta)   
        return None
    
    

def select_action(Gsys):
    """
    Use Thompson Sampling to select an action. 
    
    Input:
      Gsys:  the game system object. 
      
    Output:
      a:   an action/arm, an integer in [K]. 
    """
    
    theta_hat = generate_parameter_sample(Gsys)
    #print('theta_hat:', theta_hat)
    
    a = opt.argmax_of_array(theta_hat) 
    #print('Actual Action:', a)
    
    return a


def generate_parameter_sample(Gsys):
    """
    Generate a sample theta_hat based on the current posterior distribution of the parameters.   
    
    Input:
      Gsys:   a game system object. 
    
    Output:
      theta_hat: a vector of values in [0,1].
    """   
    
    return np.random.beta(Gsys.state.Alpha, Gsys.state.Beta)
     
