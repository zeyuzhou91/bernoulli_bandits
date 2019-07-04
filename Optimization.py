import numpy as np
import scipy as sp
import scipy.stats as st


def argmax_of_array(array):
    """
    Find the index of the largest value in an array of real numbers. 
    
    Input:
      array:  an array of real numbers. 
    
    Output:
      index:  an integer in [K], where K = len(array)
    """
    
    return np.argmax(array)