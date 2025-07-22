import numpy as np
import pandas as pd
from typing import Optional

class ConformalPcontrol:
    def __init__(self, q_0, alpha=0.15, beta=0.1, n=1):
        self.alpha = alpha
        self.beta = beta
        self.q = q_0
        self.S = []
        self.E = []
        self.C = None
        self.eta = None
        self.n = n  # Dimension of the model, used for scaling if needed
        
    def compute_interval(self, meu, sigma) -> Optional[pd.Interval]:
        lower_bound = (meu - self.q * sigma).item()
        upper_bound = (meu + self.q * sigma).item()
        # self.C = [pd.Interval(l.item(), u.item(), closed='both') for l, u in zip(lower_bound, upper_bound)]
        if lower_bound > upper_bound:
            print("Warning: Lower bound is greater than upper bound. Adjusting bounds.")
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            print(f"Meu: {meu}, Sigma: {sigma}, q: {self.q}")
            raise ValueError("Lower bound cannot be greater than upper bound.")
        try:
            self.C = pd.Interval(left=lower_bound, right=upper_bound, closed='both') # 'both', 'left', 'right', or 'neither'
        except Exception as e:
            print(f"Error occurred while creating interval: {e}")
            print(f"Meu: {meu}, Sigma: {sigma}, q: {self.q}")
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            raise e
        return self.C
    
    def compute_error(self,obs):
        # e_t = [xi in interval for interval, xi in zip(self.C, obs)]
        e_t = 1- int(obs in self.C)
        self.E.append(e_t)
        print(f"e_t:{e_t}, mean_error = {np.mean(self.E)}")
        # if len(e_t) == 1:
        #     e_t = 1- int(e_t[0])
        return e_t
    
    def compute_score(self, obs, meu, sigma):
        if self.n == 1:
            # For 1D case, sigma is a scalar
            s_t = np.abs(obs - meu) / sigma
        else:
            #sigma here is a covariance matrix
            s_t = np.sqrt((obs - meu).T @ np.linalg.inv(sigma) @ (obs - meu))
        self.S.append(s_t)
        return s_t
    
    def compute_eta(self) -> Optional[float]:
        self.eta = self.beta * np.max(self.S) if self.S else 0
        return self.eta
    
    def compute_quantile(self):
        mean_last_k = np.mean(self.E[-10:])
        print(f"mean_last_k: {mean_last_k}")
        q_t1 = self.q + self.eta * (mean_last_k - self.alpha)
        
        q_t1 = max(q_t1, 0.001)  # Ensure q does not go below a threshold
        self.q = q_t1
        return self.q
    
    


