from typing import Any, Optional, Union
import numpy as np
from filter.NESDE import ESDE
import torch 


class LinearModel_Estimator():
    def __init__(
        self, 
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        R_cost: np.ndarray,
        Q_cost: np.ndarray,
        x_goal: np.ndarray,
        time_step: float,
        max_episode_time_steps: int,
        device: Optional[Union[torch.device,str]],
        initial_mean: np.ndarray = np.array([0, 0]),
        initial_cov: np.ndarray = np.diag([0,0])
    ):
        self.state_dim = A.shape[0]
        self.A =  A
        self.B =  B
        self.H = H
        self.Q = Q   
        self.R = R
        self.R_cost = R_cost
        self.Q_cost = Q_cost
        self.x_goal = x_goal
        self.time_step = time_step
        self.max_episode_time = max_episode_time_steps * time_step

        self._last_obs = None
        self._last_var = None
        self.control_actions = None
        self.sample_actions = None
        self.device = device
       
        self.esde = ESDE(device=self.device, n=self.state_dim)
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        self.params = {'V': eigenvectors, 'lambda': eigenvalues, 'Q': self.Q}
        self.dist_estimates = None

        self.initial_mean = initial_mean
        self.initial_cov = initial_cov
    
    def get_dimention(self) -> int:
        """
        Returns the dimension of the model.
        """
        return self.state_dim
    
    def reset(self, seed=None, options=None):
        try:
            self.control_actions = []
            self.sample_actions = []
            self.dist_estimates = []

            self._last_obs = self.initial_mean.flatten()
            self._last_var = self.initial_cov
            # self._last_full_obs = np.vstack((self._last_obs.reshape(-1, 1), self._last_var.reshape(-1, 1)))
            self.dist_estimates.append({'mean': self._last_obs, 'var': self._last_var})

        
        except Exception as e:
            import traceback
            print("❌ Exception in LinearModel.reset():")
            traceback.print_exc()
            raise e

    def predict(self, action):
        Bu = (self.B @ action).reshape(-1, 1)  # control input in the form of Bu
        next_obs, next_var = self.esde.predict_an(
            self._last_obs, self._last_var, Bu,
            np.array([self.time_step]), self.params
        )
        next_obs = next_obs.flatten() #since next_obs is (1, state_dim), and we want (state_dim,)
        next_var = next_var[0] #remove the batch dimension
        self.dist_estimates.append({'mean': next_obs, 'var': next_var})
        self._last_obs = next_obs
        self._last_var = next_var
            

        return self._last_obs, self._last_var
    
    def update(self, obs):
        if self._last_obs is None:
            raise ValueError("Model has not been reset. Call predict first")
        mask = np.array([True] * self.state_dim)  # Assuming all states are observed
        cond_dist_mu, cond_dist_sigma = self.esde.conditional_dist(
                    self._last_obs, self._last_var,
                    mask, obs.flatten(), self.R
                )
        print(f"Conditional distribution mean: {cond_dist_mu}, obs: {obs.flatten()}, conditional variance: {cond_dist_sigma}")
        self.dist_estimates.pop()  # Remove the last estimate (which came from the predict step) before appending the new one
        self.dist_estimates.append({'mean': cond_dist_mu, 'var': cond_dist_sigma})

        self._last_obs = cond_dist_mu
        self._last_var = cond_dist_sigma.reshape(self.state_dim, self.state_dim)
        return cond_dist_mu, cond_dist_sigma
        
    def get_param(self, str_param: str) -> Any:
        if hasattr(self, str_param):
            return getattr(self, str_param)
        else:
            raise ValueError(f"Parameter '{str_param}' not found in the model.")
    
    def update_params(self, A=None, B=None, Q=None, H=None, R=None):
        self.Q = Q
        if A is not None:
            self.A = A
            eigenvalues, eigenvectors = np.linalg.eig(self.A)
            self.params['lambda'] = eigenvalues
            self.params['V'] = eigenvectors
        if B is not None:
            self.B = B
        if H is not None:
            self.H = H  
        if R is not None:
            self.R = R
        if Q is not None:
            self.Q = Q
            self.params['Q'] = Q
        
        
    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
