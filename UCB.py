import numpy as np
import auxiliary as aux

class State:
    def __init__(self, K):
                
        self.N = np.zeros(K)  # N[i] is the number of times arm i has been played so far
        self.S = np.zeros(K)  # S[i] is the total reward obtained from arm i so far
        self.avg = np.zeros(K)  #avg[i] is the average reward obtained from arm i so far
        self.I = np.ones(K) * float('inf')  # I[i] is the index value of arm i 
    
    def update(self, a, obs, t):
        """
        Update the state variables based on action a and observation obs. 
        
        Input:
          a:    the action/arm taken in round t, an integer in [K]
          obs:  the observation incurred in round t, 0 or 1
        """
        
        self.N[a] += 1
        self.S[a] += obs
        self.avg[a] = self.S[a] / float(self.N[a])
        
        for i in range(len(self.I)):
            if self.N[i] == 0:  # this arm has not been pulled
                pass
            else:
                self.I[i] = self.avg[i] + np.sqrt(2*np.log(t+1)/self.N[i])
    
    def print(self):
        """
        Print the current value of the state variables."
        """
        
        print('N = ', self.N)
        print('S = ', self.S)
        print('avg = ', self.avg)
        print('I = ', self.I)   
        return None
    
    
def select_action(Gsys):
    """
    Use the UCB (Upper Confidence Bound) algorithm to select an action. 
    
    Input:
      Gsys:  the game system object. 
      
    Output:
      a:   an action/arm, an integer in [K]. 
    """
    
    a = aux.argmax_of_array(Gsys.state.I) 
    
    return a