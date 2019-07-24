import numpy as np
import scipy as sp
import scipy.stats as st


def argmax_of_array(array):
    """
    Find the index of the largest value in an array of real numbers. In the case 
    where there are more than one largest values, randomly choose one of them. 
    
    Input:
      array:  an array of real numbers. 
    
    Output:
      index:  an integer in [K], where K = len(array)
    """
    
    ## Method 1.
    ## This method supports random selection in the case of more than one largest values. 
    #max_val = np.max(array)
    #max_indices = np.where(array == max_val)[0]
    #np.random.shuffle(max_indices)
    #ind = max_indices[0]
    
    # Method 2.
    # Simple but does not support random selection in the case of more than one largest values. 
    ind = np.argmax(array)
    
    return ind