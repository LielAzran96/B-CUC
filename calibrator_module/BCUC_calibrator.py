import numpy as np
from models.linear_model import LinearModel_Estimator
from calibrator_module.conformalPcontrol import ConformalPcontrol
import copy

class BCUC_Calibrator:
    def __init__(
            self, 
            initial_model: LinearModel_Estimator, 
            beta: float = 0.1, 
            alpha: float = 0.15, 
            gamma = 0.1, 
            eta_max = 0.5,
            ):
        self.model = copy.deepcopy(initial_model)
        self.beta = beta
        self.alpha = alpha
        self.q = np.array([2.0]) # initial quantile
        self.Q = self.model.get_param('Q') # initial Q matrix
        self.n = self.model.get_dimention() # dimention of the model
        self.lam = 0.1  # lambda for updating Q matrix
        self.conformal_p_control = ConformalPcontrol(alpha=self.alpha, 
                                                     beta=self.beta, 
                                                     q_0=self.q,  
                                                     n=self.n, 
                                                     eta_max=eta_max, 
                                                     gamma=gamma)
        


    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs = True):
        """
        Calibrate the model using the provided observations.
        
        Parameters:
        - observations: np.ndarray of shape (n_timesteps, n_observations, n_features))
        
        Returns:
        - calibrated_model: LinearModel
        """
        # n_samples, n_observations = observations.shape
 
        self.interval_widths = []  # Store interval widths for analysis
        self.Q_history = []
        self.q_history = []
        self.meu_history = []
        
        if self.n == 1:
            n_samples, n_observations = observations.shape
            n_features = 1
        else:
            n_samples, n_observations ,n_features = observations.shape
            
        if n_features != self.Q.shape[0]:
            raise ValueError("Number of features in observations must match the model's Q matrix dimensions.")
        
        if actions is None:
            actions = np.zeros((n_samples, n_observations, n_features))
        if reset_obs:
            # Reset the model to its initial state
            self.model.reset()
        counter = 0   
        for obs, action in zip(observations, actions):
            self.Q_history.append(self.Q.flatten().copy())
            self.q_history.append(self.q)
            meu, cov = self.model.predict(action)
            self.meu_history.append(meu.flatten().copy())
            
            if self.model.state_dim == 1:
                    sigma = np.sqrt(cov)
            else:
                    sigma = cov
            # Compute the conformal prediction interval
            self.interval_widths.append(2*sigma.flatten())

            if not np.isnan(obs).any():
                counter += 1
                
                self.conformal_p_control.compute_score(obs, meu, sigma)
                self.conformal_p_control.compute_interval(meu, sigma)
                self.conformal_p_control.compute_error(obs)
                self.conformal_p_control.compute_eta()  
                if counter % 1 == 0:
                    self.q = self.conformal_p_control.compute_quantile(self.q)
                    # Q_new = self.conformal_p_control.compute_quantile(self.Q)
                    # Update the model's Q matrix based on the calibrated quantile
                    Q_new  = np.pow(self.q, 2) * self.Q
                 
                    # updated_Q = (1 - self.lam) * self.Q + self.lam * Q_new
                    self.model.update_params(Q = Q_new)
                    self.Q = self.model.get_param('Q')

                #   # if counter % 5 == 0:
                # self.q = self.conformal_p_control.compute_quantile()

                # # # Update the model's Q matrix based on the calibrated quantile
                # self.model.update_params(Q = np.pow(self.q, 2) * self.Q)
                # self.Q = self.model.get_param('Q')
                
                print(f"Updated Q : {self.Q}, q: {self.q}")  
                self.model.update(obs)
                
        return self.model
                
    def get_calibrated_model(self) -> LinearModel_Estimator:
        return self.model
            