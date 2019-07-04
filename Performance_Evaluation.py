import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy as sp
import scipy.stats as st


def calculate_scores(Gsys):
    """
    Measure the performance as recorded in Gsys. 
    
    Input:
      Gsys:    a game system object. 
    """    
    scores = np.zeros(Gsys.T)
    
    # accumulative
    for t in range(Gsys.T):
        if t == 0:
            scores[t] = Gsys.regs[t] 
        else:
            scores[t] = scores[t-1] + Gsys.regs[t]
            
    ## running average
    #for t in range(Gsys.T):
        #scores[t] = scores[t] / float(t+1)   
    
    ## a different metric
    #for t in range(Gsys.T):
        #if (Gsys.A[t] == Gsys.best_action).all():
            #scores[t] = 1.0
        #else:
            #scores[t] = 0.0
    
    return scores
    

def myplot(ydata, yname, title):
    """
    Plotting. 
    
    Input:
      ydata:  a numpy array or list of values. 
      yname:  a string. the name of the y-axis. 
      title:  a string. the title. 
    """
    
    T = len(ydata)
    
    plt.figure(1)
    plt.plot(range(T), ydata)
    # plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(yname)
    #plt.savefig('figs/beta_distributions.png')
    plt.title(title)
    plt.xlim(0, T*1.1)
    plt.ylim(0, T*1.1)
    plt.show()    
    
    return None    
    



        
if __name__ == "__main__":
            
    
    N = 8
    K = 4
    x = [12, 7, 6, 1, 3, 4, 5, 8]
    mytm = 'B'
    othertm_sel = [1, 3]
    tot_rank = calculate_my_best_possible_total_rank(x, mytm, othertm_sel, N, K)