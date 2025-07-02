import numpy as np
import pandas as pd

class ConformalPcontrol:
    def __init__(self, alpha=0.05, beta=0.1, q_0):
        self.alpha = alpha
        self.beta = beta
        self.q = q_0
        self.S = []
        self.E = []
        self.C = None
        self.eta = None
        
    def compute_interval(self, meu, sigma):
        lower_bound = meu - self.q * sigma
        upper_bound = meu + self.q * sigma
        self.C = pd.Interval(left=lower_bound, right=upper_bound, closed='both') # 'both', 'left', 'right', or 'neither'
        return lower_bound, upper_bound
    
    def compute_error(self,obs):
        e_t = int(obs in self.C)
        self.E.append(e_t)
        return e_t
    
    def compute_score(self, obs, meu, sigma):
        s_t = np.abs(obs - meu) / sigma
        self.S.append(s_t)
        return s_t
    
    def compute_eta(self):
        self.eta = self.beta * np.max(self.S) if self.S else 0
        return self.eta
    
    def compute_quantile(self):
        q_t1 = self.q + self.eta * (np.mean(self.E) - self.alpha)
        self.q = q_t1
        return self.q
    
    


