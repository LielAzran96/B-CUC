from typing import Any, Optional, Union
import numpy as np
from NESDE import ESDE
import torch 
from debugs_utils import conditional_print


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
        state_dim = A.shape[0]

        self._last_obs = None
        self._last_var = None
        self.control_actions = None
        self.sample_actions = None
        self.device = self.env.device if hasattr(self.env, "device") else None
       
        self.esde = ESDE(device=self.device, n=self.state_dim)
        eigenvalues, eigenvectors = np.linalg.eig(self.env.A)
        self.params = {'V': eigenvectors, 'lambda': eigenvalues, 'Q': self.env.Q}
        self.dist_estimates = None

        self.initial_mean = initial_mean
        self.initial_cov = initial_cov
        
    def reset(self, seed=None, options=None):
        try:
            obs_from_original_env, _ = self.env.reset()
            self.control_actions = []
            self.sample_actions = []
            self.dist_estimates = []

            self._last_obs = self.initial_mean.flatten()
            self._last_var = self.env.initial_cov
            # self._last_full_obs = np.vstack((self._last_obs.reshape(-1, 1), self._last_var.reshape(-1, 1)))
            self.dist_estimates.append({'mean': self._last_obs, 'var': self._last_var})

        
        except Exception as e:
            import traceback
            print("❌ Exception in LinearEnvWrapper_LQG.reset():")
            traceback.print_exc()
            raise e

    # def step(self, sample) -> tuple:
    #     try:
    #         a_t = self.LQG.choose_action(self._last_obs)
    #         self.control_actions.append(a_t)

    #         obs_from_original_env, reward, terminated, done, info = self.env.step(a_t)

    #         state_estimate, var_estimate = self.esde.predict_an(
    #             self._last_obs, self._last_var, self.env.Bu,
    #             np.array([self.env.time_step]), self.params
    #         )
    #         state_estimate = state_estimate.flatten()
    #         var_estimate = var_estimate[0]  # Remove the batch dimension

    #         if sample:
    #             cond_dist_mu, cond_dist_sigma = self.esde.conditional_dist(
    #                 state_estimate, var_estimate,
    #                 np.array([True, True]), obs_from_original_env.flatten(), self.env.R
    #             )
    #             cond_dist_mu = cond_dist_mu.reshape(self.env.state_dim, 1)
    #             cond_dist_sigma = cond_dist_sigma.reshape(self.env.state_dim * self.env.state_dim, 1)
    #             conditional_print(f"cond_dist_sigma shape: {cond_dist_sigma.shape}, cond_dist_sigma: {cond_dist_sigma}", False)

    #             self.dist_estimates.append({'mean': cond_dist_mu, 'var': cond_dist_sigma})
    #             obs = np.vstack((cond_dist_mu, cond_dist_sigma))

    #             self._last_obs = cond_dist_mu
    #             self._last_var = cond_dist_sigma.reshape(self.env.state_dim, self.env.state_dim)
    #         else:
    #             state_estimate = state_estimate.reshape(self.env.state_dim, 1)
    #             var_estimate = var_estimate.reshape(self.env.state_dim * self.env.state_dim, 1)

    #             self.dist_estimates.append({'mean': state_estimate, 'var': var_estimate})
    #             obs = np.vstack((state_estimate, var_estimate))

    #             self._last_obs = state_estimate
    #             self._last_var = var_estimate.reshape(self.env.state_dim, self.env.state_dim)

    #         extra_info = {
    #             "dist_estimate": self.dist_estimates[-1],
    #             "state": obs_from_original_env,
    #             "control_action": a_t,
    #             "sample_action": sample
    #         }
    #         info.update(extra_info)

    #         return obs, reward, terminated, done, info

        # except Exception as e:
        #     import traceback
        #     print("❌ Exception in LinearEnvWrapper_LQG.step():")
        #     traceback.print_exc()
        #     raise e

    def predict(self, action):
        
        if self._last_obs is None or self._last_var is None:
            self._last_obs = self.env.initial_mean.reshape(self.env.state_dim, 1) - self.env.x_goal
            self._last_var = self.env.initial_cov

        Bu = (self.env.B @ action).reshape(-1, 1)  # control input in the form of Bu
        next_obs, next_var = self.esde.predict_an(
            self._last_obs, self._last_var, Bu,
            np.array([self.env.time_step]), self.params
        )
        next_obs = next_obs.flatten() #since next_obs is (1, state_dim), and we want (state_dim,)
        next_var = next_var[0] #remove the batch dimension
        self.dist_estimates.append({'mean': next_obs, 'var': next_var})
        self._last_obs = next_obs
        self._last_var = next_var
        
        return next_obs, next_var
    
    def update(self, obs):
        if self._last_obs is None:
            raise ValueError("Model has not been reset. Call predict first")
        cond_dist_mu, cond_dist_sigma = self.esde.conditional_dist(
                    self._last_obs, self._last_var,
                    np.array([True, True]), obs.flatten(), self.R
                )

        self.dist_estimates.append({'mean': cond_dist_mu, 'var': cond_dist_sigma})

        self._last_obs = cond_dist_mu
        self._last_var = cond_dist_sigma.reshape(self.env.state_dim, self.env.state_dim)
        return cond_dist_mu, cond_dist_sigma
        
    def getQ(self):
        return self.Q    

    def setQ(self, Q):
        self.Q = Q
        

    # def calc_covariance_noise_matrix(self, state, action: Optional[np.ndarray] = None):
    #     ranges = [(1, 1.5)]
    #     matrix_index = None
    #     state_mag = np.linalg.norm(state)
    #     for i, (low, high) in enumerate(ranges):
    #         if low <= state_mag < high:
    #             matrix_index = 5
    #             break
    #     if matrix_index is None:
    #         matrix_index = 1
    #     new_Q = self.env.original_Q * matrix_index
    #     self.env.Q = new_Q
    #     return new_Q

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
