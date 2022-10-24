import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from entropy_complexity import entropy_complexity, get_borders
from tqdm import tqdm
from math import factorial



def functions_testing(inp, checker):
    if checker == 0:
        return 0
    
    elif checker == 1:
        n = (10**(len(str(inp)))) * factorial(inp)
        
    elif checker == 2:
        n = int(factorial(inp) * np.exp((inp**((1.055+inp/100)))))
        
    elif checker == 3:
        n = int(factorial(inp) * np.exp(inp))
        
    
    return n
    
    
def plotting_test(s=2, inp=0, checker=0):
    for i in range(s,inp):
        length = functions_testing(i, checker)
        noise = np.random.normal(size=length)
        M = 1
        min_ec, max_ec = get_borders(n=i, m=M)
        EC = entropy_complexity(noise, n=i, m=M)
        print(EC)      
        f, ax = plt.subplots(1,1,figsize=(16, 7))

        ax.plot(max_ec[:,0], max_ec[:,1],color='r')
        ax.plot(min_ec[:,0], min_ec[:,1],color='r')
        counter = 0
        ax.scatter(*EC, label="Noise for function %s" %checker, s=200, marker='.')
           
    # ax.scatter(*EC[-2], label=ts_names[-2], s=200, marker='^')
    # ax.scatter(*EC[-1], label=ts_names[-1], s=200, marker='s')


        ax.set_xlabel('entropy, $H$')
        ax.set_ylabel('complexity, $C$')
        plt.title("Normalized Entropy-Complexity plane, N=%s" % (i), fontsize = 17)
        ax.legend()
        plt.savefig("Fig_%s" % (i-2))
        
        
plotting_test(11, 20, 1)