import numpy as np
from typing import Optional, Union
import torch
from filter.NESDE import ESDE

class LinearModel():
    '''
    Linear model for generate data, with gaussian noise for both the processes and measurement.
    @input params -
    A - dynamics matrix
    B - control matrix
    H - observation matrix
    Q - process noise matrix
    R - measurement noise matrix

    '''
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

        self.original_Q = self.Q
        self.R = R
        self.R_cost = R_cost
        self.Q_cost = Q_cost
        self.x_goal = x_goal
        self.time_step = time_step
        self.state = None       
        self.time = None
        self.ts = np.linspace(0, 1/time_step, int(1/time_step) + 1)
        self.u = None
        self.initial_mean = initial_mean
        self.initial_cov = initial_cov
        self.device = device
        self.esde = ESDE(device=self.device, n=self.state_dim)
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        self.params = {'V':eigenvectors, 'lambda': eigenvalues, 'Q': self.Q}
        
        # Initialize tracking variables
        self.states = []
        self.counter = 0
        
    def step(self, action):
        self.counter += 1
        x_t = self.state
        self.u = action
        cov = np.diag(np.ones(self.state_dim) * 0.0001)  # small process noise covariance
        # cov = np.diag([0.0001, 0.0001])  # small covariance for the noise

        self.Bu = (self.B @ self.u).reshape(-1, 1)  # control input in the form of Bu
        mean, cov = self.esde.predict_an(x_t, cov, self.Bu, np.array([self.time_step]), self.params)
        cov = cov[0]
        x_t1 = np.random.multivariate_normal(mean.flatten(), cov)
        self.states.append(x_t1)
        self.state = x_t1
        self.time += self.time_step
        observation = self._get_noisy_observation()

        return observation

    def _get_noisy_observation(self):
        """Generate noisy observation from current state"""
        v_t = np.random.multivariate_normal(np.zeros(self.H.shape[0]), self.R)
        o_t = self.H @ self.state.flatten() + v_t
        return o_t.flatten()


    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return seed

    def reset(self):
        """Reset the environment to initial state"""
        self.state = np.random.multivariate_normal(self.initial_mean, self.initial_cov).flatten()
        self.time = 0.0
        self.states=[]
        self.states.append(self.state)
        self.counter = 0
        
        # Get initial observation
        # initial_obs = self._get_noisy_observation()
        return self.state
