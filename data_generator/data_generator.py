import numpy as np
from typing import Optional, Union
import torch
from NESDE import ESDE

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
        # self.A = time_step * A

        self.B =  B
        self.H = H
        self.Q = Q   

        self.original_Q = self.Q
        self.R = R
        self.R_cost = R_cost
        self.Q_cost = Q_cost
        self.x_goal = x_goal
        self.time_step = time_step
        self.max_episode_time = max_episode_time_steps*time_step
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
        
        


        
    def step(self, action):
        self.counter += 1
        x_t = self.state
        self.u = action
        cov = np.diag([0.0001, 0.0001])  # small covariance for the noise

        self.Bu = (self.B @ self.u).reshape(-1, 1)  # control input in the form of Bu
        mean, cov = self.esde.predict_an(x_t, cov, self.Bu, np.array([self.time_step]), self.params)
        cov = cov[0]
        conditional_print(f"cov shape: {cov.shape}, cov: {cov}", False)
        x_t1 = np.random.multivariate_normal(mean.flatten(), cov).reshape(self.state_dim, 1)
        self.states.append(x_t1)
        self.state = x_t1

        done = bool(self.time > self.max_episode_time)
        self.time += self.time_step

        cost = self._quadratic_cost(action)
        observation = self._get_noisy_observation()

        return observation, done, {}

    def _get_noisy_observation(self):
        v_t = np.random.multivariate_normal(np.zeros(self.state_dim), self.R).reshape(self.state_dim, 1)
        o_t = self.H @ self.state + v_t
        return o_t

    def reset(self, seed=None, options=None):
        self.counter = 0
        self.states = []
        self.time = 0

        state = np.random.multivariate_normal(self.initial_mean, self.initial_cov).reshape(self.state_dim, 1)
        self.state = state - self.x_goal
        self.states.append(self.state)

        return self.state, {}

    # def _quadratic_cost(self, action):
    #     x_t = self.state
    #     x_cost = x_t.T @ self.Q_cost @ x_t
    #     u_cost = action.T @ self.R_cost @ action
    #     return float(x_cost + u_cost)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return seed
   
   