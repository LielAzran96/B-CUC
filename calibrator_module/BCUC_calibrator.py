import numpy as np
from models.linear_model import LinearModel_Estimator
from conformalPcontrol import ConformalPcontrol
import copy

class BCUC_Calibrator:
    def __init__(self, initial_model: LinearModel_Estimator, beta: float = 0.1, alpha: float = 0.05):
        self.model = copy.deeptcopy(initial_model)
        self.beta = beta
        self.alpha = alpha
        self.q_0 = 1.0
        self.Q = self.initial_model.getQ()
        self.n = self.initial_model.get_dimention() # dimention of the model
        self.conformal_p_control = ConformalPcontrol(alpha=self.alpha, beta=self.beta, q_0=self.q_0)
        
     
    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None):
        """
        Calibrate the model using the provided observations.
        
        Parameters:
        - observations: np.ndarray of shape (n_timesteps, n_observations, n_features))
        
        Returns:
        - calibrated_model: LinearModel
        """
        n_samples, n_observations, n_features = observations.shape

        # if self.n == 1:
        #     n_samples, n_observations, n_features = observations.shape
        #     # n_features = 1
        # else:
        #     n_samples, n_observations ,n_features = observations.shape
            
        if n_features != self.Q.shape[0]:
            raise ValueError("Number of features in observations must match the model's Q matrix dimensions.")
        
        if actions is None:
            actions = np.zeros((n_samples, n_observations, n_features))
        
        for obs, action in zip(observations, actions):
            meu, cov = self.model.predict(obs, action)
            if obs is not None:
                # CONTINUE FROM HERE
                break