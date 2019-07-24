import numpy as np
import scipy as sp
import scipy.stats as st
import auxiliary as aux


class State:
    def __init__(self, K, Npar):
        
        self.K = K  # number of arms
        self.Npar = Npar  # number of particles
        
        # the set of particles
        self.Theta = generate_particles(K, Npar)
        
        # the weights associated with the particles
        # initialized to be all having equal weights
        self.W = np.ones(Npar) * (1.0/Npar)   
        
    
    def update(self, a, obs):
        """
        Update the state variables (the weights).  
        
        Input:
          a:    the action/arm taken in round t, an integer. 
          obs:  the observation incurred in round t, 0 or 1. 
        """
        
        new_w = np.zeros(self.Npar)  # the unnormalized new weight vector
        for k in range(self.Npar):
            lh = calculate_likelihood(self.Theta[k][int(a)], obs)
            #print('likelihood =', lh)
            new_w[k] = lh * self.W[k]
        new_W = 1.0/(np.sum(new_w)) * new_w   # normalizing
        #print('new_W =', new_W)
        self.W = new_W
        
        return None
    
    
    def print(self):
        """
        Print the current value of the state variables."
        """
        
        print('Particle weights = ', self.W)   
        return None


    
def select_action(Gsys):
    """
    Use particle filter to select an action. 
    
    Input:
      Gsys:  the game system object.  
      
    Output:
      a:   an action/arm, an integer.
    """
    
    theta_hat = generate_parameter_sample(Gsys)
    #print('theta_hat:', theta_hat)
    
    a = aux.argmax_of_array(theta_hat) 
    #print('Actual Action:', a)
    
    return a    



def generate_particles(K, Npar):
    """
    Generate the set of particles. 
    
    Input:
      K:    number of arms
      Npar: number of particles, an integer 
    
    Output:
      Par:  the set of particles, a numpy array. Par[0] is the first particle (of appropriate dimension). 
    """
    
    Par = np.zeros((Npar, K)) 
    
    # Method 1: Each particle is a dimension-K vector. We generate each particle 
    # uniformly at random from the space [0,1]^K. 
    # This method for any integer K.
    Par = np.random.uniform(0, 1, (Npar, K))
    #print("The set of particles is: ", Par)
    
    
    # Method 2: We generate m points on [0,1] uniformly at random and let the set
    # of particles be the K-fold meshgrid of these m points. 
    # E.g. If m = 3 and the points are [0.1, 0.4, 0.7]. 
    # Then for K = 2, the particles are [0.1, 0.1], [0.1, 0.4], [0.1, 0.7],
    # [0.4, 0.1], [0.4, 0.4], [0.4, 0.7], [0.7, 0.1], [0.7, 0.4], [0.7, 0.7].
    # This method requires Npar = m^K.
    

    ## Method 3: Pre-determined points. We pre-determine the particles for each parameter as a product set 
    ## of np.linspace(0, ub, N) for some suitable N. 
    ## ZEYU: modify this part. 
    #N = round(Npar**(1.0/d))
    #x = np.linspace(0, ub, N)
    #OneSet = np.stack(np.meshgrid(x,x,x), axis=-1).reshape(-1,d)  # WARNING: only applies to the case d=3
    #for i in range(M):
        #for j in range(B[i]):
            #Par[i][j] = OneSet
    ##print("The set of particles is: ", Par)
            
    return Par   
    


def generate_parameter_sample(Gsys):
    """
    Generate a sample theta_hat (one particle) based on the current weights on the particles. 
    
    Input:
      Gsys:   a game system object. 
    
    Output:
      theta_hat: a length-K vector of values in [0,1].
    """   
    
    theta_hat = np.zeros(Gsys.K) 
    k = np.random.choice(Gsys.Npar, 1, p=Gsys.state.W)[0]  # np.random.choice outputs an array
    theta_hat = Gsys.state.Theta[k]   
    # ZEYU: is there a quicker way to implement this?
    
    return theta_hat




def calculate_likelihood(theta, obs):
    """
    Calculate the likelihood/probability of observing obs, if the parameter is theta. 
    
    Input:
      theta:  a probability. 
      obs:    0 or 1. 
    
    Output:
      lh:     a number in [0,1], the likelihood/probability. 
    """
    
    if obs == 1:
        lh = theta
    else:
        lh = 1-theta
    
    return lh


if __name__ == "__main__":
    pass