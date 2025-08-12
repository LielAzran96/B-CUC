


import numpy as np
from models.linear_model import LinearModel_Estimator
from calibrator_module.conformalPcontrol_finalVersion import ConformalPcontrol
import copy
from scipy.stats import norm
from collections import deque

class DirectQBCUCCalibrator:
    def __init__(
            self, 
            initial_model: LinearModel_Estimator, 
            beta: float = 0.3, 
            alpha: float = 0.32, #for matching gaussian coverage and making Q_t converge to real Q* 
            gamma: float = 0.5, 
            eta_max: float = 0.3,
            lambd: float = 0.01,  # lambda for updating Q matrix,
            update_after_every :int = 15,  # update Q after every n observations (default 15, can be changed)
        ):
        self.model = copy.deepcopy(initial_model)
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.eta_max = eta_max
        self.lambd = lambd
        self.initial_q = np.array([1.0])  #not really used
        self.initial_Q = self.model.get_param('Q')  # initial Q matrix
        self.n = self.model.get_dimention()  # model dimension
        self.update_after_every = update_after_every  # update Q after every n observations


        # === Anchored Q coupling settings (for quantile_coupled mode) ===
        self.Q_min, self.Q_max = 1e-8, 1e6



    # ---------- helpers ----------
    def _proj_Q(self, Q):
        return np.clip(Q, self.Q_min, self.Q_max)

    def _update_Q(self, Q_new):
        
        Q_new = self.lambd* Q_new + (1-self.lambd) * self.Q  # lambda for smoothing
        
        return self._proj_Q(Q_new)


    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        # reset run logs
        self.interval_widths = []
        self.Q_history = []
        self.q_history = []
        self.meu_history = []
        self.nll_history = []
        # reset model
        self.Q = self.initial_Q
        self.q = self.initial_q
        self.conformal_p_control = ConformalPcontrol(alpha=self.alpha, 
                                                beta=self.beta, 
                                                q_0=self.q,  
                                                n=self.n, 
                                                eta_max=self.eta_max, 
                                                gamma=self.gamma)
        # shape checks
        if self.n == 1:
            n_samples, n_observations = observations.shape
            n_features = 1
        else:
            n_samples, n_observations, n_features = observations.shape

        if n_features != self.Q.shape[0]:
            raise ValueError("Number of features in observations must match the model's Q matrix dimensions.")

        if actions is None:
            actions = np.zeros((n_samples, n_observations, n_features))
        if reset_obs:
            self.model.reset()

        counter = 0
        for obs, action in zip(observations, actions):
            self.Q_history.append(self.Q.flatten().copy())
            self.q_history.append(self.q)
            meu, cov = self.model.predict(action)
            self.meu_history.append(meu.flatten().copy())
            # After predict(), before quantile correction:
    

            # stds for width & metrics
            if self.model.state_dim == 1:
                std_vec = np.array([float(np.sqrt(cov))])  # 1D -> vector of length 1
                var_vec = np.array([float(cov)])
            else:
                if cov.ndim == 2:
                    var_vec = np.diag(cov)
                else:
                    var_vec = np.asarray(cov).reshape(-1)
                var_vec = np.clip(var_vec, 1e-12, None)
                std_vec = np.sqrt(var_vec)

            # width log (vector 2*sigma), and we'll aggregate later
            self.interval_widths.append(2.0 * std_vec.copy())

            # NLL (before updates) if obs available
            if not np.isnan(obs).any():
                y_vec = np.asarray(obs).reshape(-1)
                mu_vec = np.asarray(meu).reshape(-1)
                resid2 = (y_vec - mu_vec) ** 2
                # Gaussian NLL per dim, sum over dims
                nll_dims = 0.5 * (np.log(2 * np.pi * var_vec) + resid2 / np.clip(var_vec, 1e-12, None))
                self.nll_history.append(float(np.sum(nll_dims)))

                counter += 1

                # --- learn q via conformal p-control (your original path) ---
                # score & interval/error computed with controller's current q
                self.conformal_p_control.compute_score(obs, meu, std_vec if self.model.state_dim==1 else cov)
                self.conformal_p_control.q = 1  # reset q before calculating the interval
                self.conformal_p_control.compute_interval(meu, std_vec, True)   # use μ ± σ (inside the class)
                self.conformal_p_control.compute_error(obs)
                self.conformal_p_control.compute_eta()

                # update q and then Q from q (anchored) every number of observations 
                if counter % self.update_after_every== 0:
                    #here we use conformal_p_control to compute Q directly instead of q.
                    Q = self.conformal_p_control.compute_quantile(self.Q) 
                    Q_new = self._update_Q(Q)
                    self.model.update_params(Q=Q_new)
                    self.Q = self.model.get_param('Q')

                self.model.update(obs)
                
        # After the loop correct Q if alpha is different than 0.32
        if self.alpha != 0.32:
            print(f"before correction, Q = {self.Q}")
            z_alpha = norm.ppf(1 - self.alpha / 2)
            Q_corrected = self.Q / (z_alpha ** 2)
            Q_corrected = self._proj_Q(Q_corrected)
            print("Corrected Q:", Q_corrected)
            self.Q = Q_corrected
            self.model.update_params(Q=self.Q)

        return self.model

    def get_calibrated_model(self) -> LinearModel_Estimator:
        return self.model

    def get_metrics(self, targetQ: np.ndarray = 0.5) -> dict:
        """
        Returns unified metrics:
          - E_bar: mean error from ConformalPcontrol.E (in quantile_free this is μ±σ; in quantile_coupled it's μ±q_tσ)
          - coverage: 1 - E_bar
          - mean_interval_width: avg scalar width sum_i 2*sigma_i (from predictions, before updates)
          - avg_nll: mean Gaussian NLL (sum over dims), computed before updates
        """
        E_bar = float(np.mean(self.conformal_p_control.E)) if len(self.conformal_p_control.E) else np.nan
        coverage = 1.0 - E_bar if len(self.conformal_p_control.E) else np.nan

        if len(self.interval_widths):
            widths_scalar = [float(np.sum(w)) for w in self.interval_widths]
            mean_width = float(np.mean(widths_scalar))
        else:
            mean_width = np.nan

        avg_nll = float(np.mean(self.nll_history)) if self.nll_history else np.nan

        return {
            "E_bar": E_bar,
            "coverage": coverage,
            "mean_interval_width": mean_width,
            "avg_nll": avg_nll,
            "final_difference": float(np.abs(self.Q - targetQ)),
            "relative_error": float(np.abs(self.Q - targetQ) / targetQ)
        }