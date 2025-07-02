from typing import Any, Optional, Union
import numpy as np
from NESDE import ESDE
import torch 
# from stable_baselines3.common.utils_lqgPPO import to_numpy
from debugs_utils import conditional_print


   
class LinearModel_Estimator():
    def __init__(self, env, sample_cost, const_noise=True):
        super().__init__(env)
        self.env = env
        self.cost = sample_cost
        state_shape = env.state_dim
        covariance_LQG_shape = env.A.size  # NumPy's equivalent of .numel()
        total_shape = state_shape + covariance_LQG_shape

    
        self._last_obs = None
        self._last_var = None
        self.const_noise = const_noise
        self.control_actions = None
        self.sample_actions = None
        self.device = self.env.device if hasattr(self.env, "device") else None

        if not hasattr(self.env, 'esde'):
            self.esde = ESDE(device=self.device, n=state_shape)
            eigenvalues, eigenvectors = np.linalg.eig(self.env.A)
            self.params = {'V': eigenvectors, 'lambda': eigenvalues, 'Q': self.env.Q}
        else:
            self.esde = self.env.esde
            self.params = self.env.params

        self.dist_estimates = None

    def reset(self, seed=None, options=None):
        try:
            obs_from_original_env, _ = self.env.reset()
            self.control_actions = []
            self.sample_actions = []
            self.dist_estimates = []

            self._last_obs = self.env.initial_mean.reshape(self.env.state_dim, 1) - self.env.x_goal
            self._last_var = self.env.initial_cov
            self._last_full_obs = np.vstack((self._last_obs.reshape(-1, 1), self._last_var.reshape(-1, 1)))
            self.dist_estimates.append({'mean': self._last_obs, 'var': self._last_var})

            return self._last_full_obs, {
                "dist_estimate": self.dist_estimates[-1],
                "state": obs_from_original_env
            }
        except Exception as e:
            import traceback
            print("❌ Exception in LinearEnvWrapper_LQG.reset():")
            traceback.print_exc()
            raise e

    def step(self, sample) -> tuple:
        try:
            a_t = self.LQG.choose_action(self._last_obs)
            self.control_actions.append(a_t)

            obs_from_original_env, reward, terminated, done, info = self.env.step(a_t)

            state_estimate, var_estimate = self.esde.predict_an(
                self._last_obs, self._last_var, self.env.Bu,
                np.array([self.env.time_step]), self.params
            )
            state_estimate = state_estimate.flatten()
            var_estimate = var_estimate[0]  # Remove the batch dimension

            if sample:
                cond_dist_mu, cond_dist_sigma = self.esde.conditional_dist(
                    state_estimate, var_estimate,
                    np.array([True, True]), obs_from_original_env.flatten(), self.env.R
                )
                cond_dist_mu = cond_dist_mu.reshape(self.env.state_dim, 1)
                cond_dist_sigma = cond_dist_sigma.reshape(self.env.state_dim * self.env.state_dim, 1)
                conditional_print(f"cond_dist_sigma shape: {cond_dist_sigma.shape}, cond_dist_sigma: {cond_dist_sigma}", False)

                self.dist_estimates.append({'mean': cond_dist_mu, 'var': cond_dist_sigma})
                obs = np.vstack((cond_dist_mu, cond_dist_sigma))

                self._last_obs = cond_dist_mu
                self._last_var = cond_dist_sigma.reshape(self.env.state_dim, self.env.state_dim)
            else:
                state_estimate = state_estimate.reshape(self.env.state_dim, 1)
                var_estimate = var_estimate.reshape(self.env.state_dim * self.env.state_dim, 1)

                self.dist_estimates.append({'mean': state_estimate, 'var': var_estimate})
                obs = np.vstack((state_estimate, var_estimate))

                self._last_obs = state_estimate
                self._last_var = var_estimate.reshape(self.env.state_dim, self.env.state_dim)

            extra_info = {
                "dist_estimate": self.dist_estimates[-1],
                "state": obs_from_original_env,
                "control_action": a_t,
                "sample_action": sample
            }
            info.update(extra_info)

            return obs, reward, terminated, done, info

        except Exception as e:
            import traceback
            print("❌ Exception in LinearEnvWrapper_LQG.step():")
            traceback.print_exc()
            raise e

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
        next_var = next_var[0]
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


    def calc_covariance_noise_matrix(self, state, action: Optional[np.ndarray] = None):
        ranges = [(1, 1.5)]
        matrix_index = None
        state_mag = np.linalg.norm(state)
        for i, (low, high) in enumerate(ranges):
            if low <= state_mag < high:
                matrix_index = 5
                break
        if matrix_index is None:
            matrix_index = 1
        new_Q = self.env.original_Q * matrix_index
        self.env.Q = new_Q
        return new_Q

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
