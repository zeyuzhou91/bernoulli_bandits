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
    
    max_val = np.max(array)
    max_indices = np.where(array == max_val)[0]
    np.random.shuffle(max_indices)
    
    return max_indices[0]