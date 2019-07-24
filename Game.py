import numpy as np
import scipy as sp
import scipy.stats as st
import Thompson_Sampling as TS
import Particle_Filter as PF
import UCB as ucb
import auxiliary as aux


class System:
    def __init__(self, K, T, Npar, algo):
        
        # number of arms
        self.K = K  
        
        # time horizon
        self.T = T  
        
        # number of particles 
        self.Npar = Npar        
        
        # algorithm
        self.algo = algo
        
        # The true theta vector of length K
        self.theta_true = np.zeros(K)        
        
        # The best action
        self.best_action = 0  
        
        # The state variables
        if algo == "Thompson Sampling":
            self.state = TS.State(K)
        elif algo == "UCB":
            self.state = ucb.State(K)
        elif algo == "Particle Filter":
            self.state = PF.State(K, Npar)
        else:
            pass
        
        # History
        ## self.X = np.zeros((T, d))  # the context history
        self.A = np.zeros(T)     # the action history
        self.OBS = np.zeros(T)    # the observation history
        self.rews = np.zeros(T)       # the reward history
        self.regs = np.zeros(T)       # The regret history       
        
        
    def update_state(self, t):
        """
        Update the staet variables. 
        
        Input: 
          t:    the round index, 0 <= t <= T-1.
        """
        
        a = int(self.A[t])
        obs = self.OBS[t]        
        if self.algo == "Thompson Sampling":
            self.state.update(a, obs)
        elif self.algo == "UCB":
            self.state.update(a, obs, t)
        elif self.algo == "Particle Filter":
            self.state.update(a, obs)
        else:
            pass
        
        return None
    
    
    def update_history(self, a, obs, rew, reg, t):
        """
        Update history.  
        
        Input:
          a:    the action taken in round t, an integer in [K]
          obs:  the observation incurred in round t, 0 or 1
          rew:  the reward obtained in round t. 
          reg:  the regret in round t.  
          t:    the round index, 0 <= t <= T-1. 
        """
        
        self.A[t] = a
        self.OBS[t] = obs
        self.rews[t] = rew
        self.regs[t] = reg
        return None  


##def generate_context(Gsys):
    ##"""
    ##Select a subset of players. 
    
    ##Input:
      ##Gsys:      the game system object. 
      
    ##Output:
      ##X:   a length-d numpy array, a context vector. 
    ##"""
    
    ##X = np.random.uniform(-1, 1, Gsys.d)
    
    ##return X
  

def generate_true_parameters(Gsys):
    """
    Generate the true theta vector. 
    
    Input:
      Gsys:   a game system object. 
    
    Output:
      theta:  a vector of values in [0,1].
    """    
    
    # Method 1: generate random values
    theta = np.random.uniform(0, 1, Gsys.K) 
    
    
    ## Method 2: use pre-determined values (for test cases)
    ## TO DO:
    
    return theta



def find_best_action(Gsys):
    """
    Find the best action/arm of the system based on the current parameters. 
    
    Input:
      Gsys:   a game system object. 
    
    Output:
      action:  an integer in [K]
    """
    
    action = aux.argmax_of_array(Gsys.theta_true)
    
    return action


def select_action(Gsys):
    """
    Select an action according to the algorithm algo. 
    
    Input:
      Gsys:  the game system object.  
      
    Output:
      a:   an action/arm, an integer in [K]. 
    """
    
    if Gsys.algo == "Thompson Sampling":
        a = TS.select_action(Gsys)
    elif Gsys.algo == "UCB":
        a = ucb.select_action(Gsys)
    elif Gsys.algo == "Particle Filter":
        a = PF.select_action(Gsys)
    else:
        pass
    
    return a
    

def play(Gsys, a):
    """
    Given action a, generate observation, record/calculate the reward and regret. 
    
    Input:
      Gsys: the game system object. 
      a:    the action, a single value. 
    
    Output:
      obs:  the observation, 0 or 1
      rew:  the reward, a single value
      reg:  the regret, a single value 
    """

    obs = obtain_observation(Gsys, a)
    rew = calculate_reward(Gsys, a)    
    reg = calculate_regret(Gsys, rew)
    
    return (obs, rew, reg)



def obtain_observation(Gsys, a):
    """
    Given an action a, generate the observation, which is random.  
    
    Input:
        a:   the action, an integer in [K] 
        
    Output:
        obs: the observation, 0 or 1
    """
    
    obs = np.random.binomial(1, Gsys.theta_true[int(a)])
    return obs



def calculate_reward(Gsys, a, obs=None):
    """
    Given an action a and the observation obs, calculate the reward.   
    
    Input:
        a:   the action, an integer in [K] 
        obs: the observation, 0 or 1 
        
    Output:
        reward: a real value.  
    """
    
    # The actual reward is usually a function of the observation. 
    # However, here we consider the expected reward, which is a function of the action. 
    # So in this code the argument obs is not used. 
    
    # Reward version 1
    reward = Gsys.theta_true[int(a)]
    
    ## Reward version 2
    #if a == Gsys.best_action:
        #reward = 1.0
    #else:
        #reward = 0.0
    
    return reward



def calculate_regret(Gsys, actual_reward):
    """
    Given an action A and the observation obs, calculate the reward.   
    
    Input:
        actual_reward: a real value. 
        
    Output:
        reg: the regret of not choosing the best action, a real value. 
    """
    
    best_reward = calculate_reward(Gsys, Gsys.best_action)
    reg = best_reward - actual_reward
    
    return reg




##def psi(u):
    
    ##return -u**2


##def noise():
    
    ##return np.random.normal(0,1)