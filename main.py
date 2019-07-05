import numpy as np
import matplotlib.pyplot as plt
import Game
##import Particle_Filter as ParFil
import Performance_Evaluation as PerfEval

np.set_printoptions(precision=4)

def run_one_simulation(algo):
    """
    Run one simulation.     
       
    Output:
       scores:  a numpy array. 
    """

    # Set up model parameters
    K = 10  # number of arms
    T = 20000 # time horizon
    # algo = "Thompson Sampling"  # the algorithm used for selecting an action
    
    # initialize an instance of the game system
    Gsys = Game.System(K, T, algo)
    
    Gsys.theta_true = Game.generate_true_parameters(Gsys)
    #print('theta_true', Gsys.theta_true)
    
    Gsys.best_action = Game.find_best_action(Gsys)
    #print('best action is', Gsys.best_action)
    
    # Game starts
    for t in range(T):
        #if t % 100 == 0:
        #    print(t) 
        
        # select an action
        a = Game.select_action(Gsys)
        # print('Action:', a)
  
        # Obtain observation, record/calculate the reward and regret  
        (obs, rew, reg) = Game.play(Gsys, a)
        #print('observation:', obs) 
        #print('reward:', rew)
        #print('regret:', reg)
        
        # Update history 
        Gsys.update_history(a, obs, rew, reg, t)
        # print('Gsys.X:', Gsys.X[:(t+1)]) 
        # print('Gsys.A:', Gsys.A[:(t+1)]) 
        # print('Gsys.R:', Gsys.R[:(t+1)]) 
        # print('Gsys.regs:', Gsys.regs[:(t+1)])   
             
        # update state variables
        Gsys.update_state(t)
        # Gsys.state.print()
        
    scores = PerfEval.calculate_scores(Gsys)   
    
    return scores


def run_simulations(num_simul, algo):
    for i in range(num_simul):
        print('simulation', i)
        if i == 0:
            p = run_one_simulation(algo)    # accumulative
        else:
            s = run_one_simulation(algo)
            p += s
    p = p / float(num_simul)
    return p    


if __name__ == "__main__":
    
    algo1 = "Thompson Sampling"
    algo2 = "UCB"
    #s1 = run_simulations(200, algo1)
    s2 = run_simulations(10, algo2) 
    
    T = len(s2)
    plt.figure(1)
    #plt.plot(range(T), s1, 'r', label = algo1)
    plt.plot(range(T), s2, 'b', label = algo2)
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('cumulative regret')
    #plt.ylabel('running average regret')
    #plt.ylabel('chance of selecting the best arm')
    #plt.title('regret = expected reward of best arm - expected reward of chosen arm')
    #plt.title('regret = 0 if chosen arm = best arm, otherwise = 1')
    #plt.xlim(0, T*1.1)
    #plt.ylim(0, T*1.1)
    #plt.savefig('figs/TS_vs_UCB.png')
    plt.show()        
