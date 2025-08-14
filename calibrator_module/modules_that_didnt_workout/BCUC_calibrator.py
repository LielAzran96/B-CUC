import numpy as np
from models.linear_model import LinearModel_Estimator
from calibrator_module.conformalPcontrol import ConformalPcontrol
import copy


import numpy as np
from models.linear_model import LinearModel_Estimator
from calibrator_module.conformalPcontrol import ConformalPcontrol
import copy


class BCUC_Calibrator:
    def __init__(
            self, 
            initial_model: LinearModel_Estimator, 
            beta: float = 0.1, 
            alpha: float = 0.32, #for matching gaussian coverage and making Q_t converge to real Q* 
            gamma: float = 0.1, 
            eta_max: float = 0.5,
            mode: str = "quantile_coupled"  # "quantile_coupled" or "quantile_free"
        ):
        """
        mode:
            "quantile_coupled" -> learn q_t with ConformalPcontrol and set Q from q (anchored)
            "quantile_free"    -> DO NOT use q; log s_t and e_t via ConformalPcontrol (with q=1),
                                  and update Q directly from coverage error in log-space.
        """
        self.model = copy.deepcopy(initial_model)
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.eta_max = eta_max
        self.mode = mode

        self.q = np.array([1.0])  # initial quantile (used only in quantile_coupled mode)
        self.Q = self.model.get_param('Q')  # initial Q matrix
        self.n = self.model.get_dimention()  # model dimension

        # Conformal controller (reused to log S and E in both modes)
        self.conformal_p_control = ConformalPcontrol(alpha=self.alpha, 
                                                     beta=self.beta, 
                                                     q_0=self.q,  
                                                     n=self.n, 
                                                     eta_max=eta_max, 
                                                     gamma=gamma)

        # === Anchored Q coupling settings (for quantile_coupled mode) ===
        self.Q_min, self.Q_max = 1e-8, 1e6
        self.Q_anchor = self.Q.copy()
        self.q_ema = float(self.q)
        self.q_ema_beta = 0.9  # smoothing for q used to set Q

        # === Quantile-free controller state ===
        # We will update Q in log-space for stability based on E_bar from ConformalPcontrol
        self.logQ = np.log(np.maximum(self.Q, self.Q_min))
        self.S_max = 0.0  # updated from ConformalPcontrol.S

        # Logs
        self.interval_widths = []   # per-step (vector) 2*std
        self.Q_history = []
        self.q_history = []
        self.meu_history = []
        self.nll_history = []       # per-step NLL (sum over dims), BEFORE updates

    # ---------- helpers ----------
    def _proj_Q(self, Q):
        return np.clip(Q, self.Q_min, self.Q_max)

    def _update_Q_from_quantile_anchor(self, q_new):
        """Anchored Q update: Q = (EMA(q)^2) * Q_anchor  (prevents compounding drift)."""
        self.q_ema = self.q_ema_beta * self.q_ema + (1.0 - self.q_ema_beta) * float(np.asarray(q_new).reshape(-1)[0])
        q_eff = max(self.q_ema, 1e-6)
        Q_new = (q_eff ** 2) * self.Q_anchor
        return self._proj_Q(Q_new)

    def _eta_from_S(self):
        # eta_t = min(eta_max, beta * max(S_1:t))
        return min(self.eta_max, self.beta * max(self.S_max, 1e-6))

    def _update_Q_direct_from_error(self, E_bar):
        """Quantile-free Q update in log-space: log Q_{t+1} = log Q_t + eta_t (E_bar - alpha)."""
        # eta_t = self._eta_from_S()
        eta_t = self.conformal_p_control.compute_eta()
        self.logQ = self.logQ + eta_t * (E_bar - self.alpha)
        Q_new = np.exp(self.logQ)
        return self._proj_Q(Q_new)
    # -----------------------------

    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        # reset run logs
        self.interval_widths = []
        self.Q_history = []
        self.q_history = []
        self.meu_history = []
        self.nll_history = []
        self.z2 = []  # for mean standardized squared error in quantile_coupled mode
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

                if self.mode == "quantile_coupled":
                    self.z2.append(np.mean((obs - meu)**2 / std_vec**2))
                    print("mean standardized squared error:", self.z2[-1])
                    # --- learn q via conformal p-control (your original path) ---
                    # score & interval/error computed with controller's current q
                    self.conformal_p_control.compute_score(obs, meu, std_vec if self.model.state_dim==1 else cov)
                    self.conformal_p_control.compute_interval(meu, std_vec)   # use μ ± q*σ (inside the class)
                    self.conformal_p_control.compute_error(obs)
                    self.conformal_p_control.compute_eta()

                    # update q and then Q from q (anchored)
                    if counter % 1 == 0:
                        self.q = self.conformal_p_control.compute_quantile(self.q)
                        Q_new = self._update_Q_from_quantile_anchor(self.q)
                        self.model.update_params(Q=Q_new)
                        self.Q = self.model.get_param('Q')

                elif self.mode == "quantile_free":
                    # --- DO NOT use q; log s_t and e_t via ConformalPcontrol with q=1 ---
                    # 1) score s_t (stored in S)
                    self.conformal_p_control.compute_score(obs, meu, std_vec if self.model.state_dim==1 else cov)

                    # 2) temporarily set q=1 to compute interval μ ± σ, then error e_t (stored in E)
                    old_q = np.copy(self.conformal_p_control.q)
                    self.conformal_p_control.q = np.array([1.0])
                    self.conformal_p_control.compute_interval(meu, std_vec)
                    self.conformal_p_control.compute_error(obs)
                    #self.conformal_p_control.compute_eta()  # optional
                    self.conformal_p_control.q = old_q

                    # 3) pull E_bar and S_max and update Q (quantile-free)
                    E_bar = self.conformal_p_control.compute_E_bar()
                    
                    #if len(self.conformal_p_control.S):
                    self.S_max = max(self.S_max, float(self.conformal_p_control.compute_S_max()))
                    # try:
                    #     self.S_max = max(self.S_max, float(self.conformal_p_control.compute_S_max()))
                    # except Exception:
                    #     self.S_max = max(self.S_max, float(np.max(np.asarray(self.conformal_p_control.S, dtype=float))))
                            
                    if counter % 1 == 0:
                        Q_new = self._update_Q_direct_from_error(E_bar)
                        self.model.update_params(Q=Q_new)
                        self.Q = self.model.get_param('Q')

                # finally assimilate the obs into the state
                self.model.update(obs)

        return self.model

    def get_calibrated_model(self) -> LinearModel_Estimator:
        return self.model

    def get_metrics(self):
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
        }




# class BCUC_Calibrator:
#     def __init__(
#             self, 
#             initial_model: LinearModel_Estimator, 
#             beta: float = 0.1, 
#             alpha: float = 0.15, 
#             gamma = 0.1, 
#             eta_max = 0.5,
#             ):
#         self.model = copy.deepcopy(initial_model)
#         self.beta = beta
#         self.alpha = alpha
#         self.q = np.array([2.0]) # initial quantile
#         self.Q = self.model.get_param('Q') # initial Q matrix
#         self.n = self.model.get_dimention() # dimention of the model
#         self.lam = 0.1  # lambda for updating Q matrix
#         self.conformal_p_control = ConformalPcontrol(alpha=self.alpha, 
#                                                      beta=self.beta, 
#                                                      q_0=self.q,  
#                                                      n=self.n, 
#                                                      eta_max=eta_max, 
#                                                      gamma=gamma)
        


#     def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs = True):
#         """
#         Calibrate the model using the provided observations.
        
#         Parameters:
#         - observations: np.ndarray of shape (n_timesteps, n_observations, n_features))
        
#         Returns:
#         - calibrated_model: LinearModel
#         """
#         # n_samples, n_observations = observations.shape
 
#         self.interval_widths = []  # Store interval widths for analysis
#         self.Q_history = []
#         self.q_history = []
#         self.meu_history = []
        
#         if self.n == 1:
#             n_samples, n_observations = observations.shape
#             n_features = 1
#         else:
#             n_samples, n_observations ,n_features = observations.shape
            
#         if n_features != self.Q.shape[0]:
#             raise ValueError("Number of features in observations must match the model's Q matrix dimensions.")
        
#         if actions is None:
#             actions = np.zeros((n_samples, n_observations, n_features))
#         if reset_obs:
#             # Reset the model to its initial state
#             self.model.reset()
#         counter = 0   
#         for obs, action in zip(observations, actions):
#             self.Q_history.append(self.Q.flatten().copy())
#             self.q_history.append(self.q)
#             meu, cov = self.model.predict(action)
#             self.meu_history.append(meu.flatten().copy())
            
#             if self.model.state_dim == 1:
#                     sigma = np.sqrt(cov)
#             else:
#                     sigma = cov
#             # Compute the conformal prediction interval
#             self.interval_widths.append(2*sigma.flatten())

#             if not np.isnan(obs).any():
#                 counter += 1
                
#                 self.conformal_p_control.compute_score(obs, meu, sigma)
#                 self.conformal_p_control.compute_interval(meu, sigma)
#                 self.conformal_p_control.compute_error(obs)
#                 self.conformal_p_control.compute_eta()  
#                 if counter % 1 == 0:
#                     self.q = self.conformal_p_control.compute_quantile(self.q)
#                     # Q_new = self.conformal_p_control.compute_quantile(self.Q)
#                     # Update the model's Q matrix based on the calibrated quantile
#                     Q_new  = np.pow(self.q, 2) * self.Q
                 
#                     # updated_Q = (1 - self.lam) * self.Q + self.lam * Q_new
#                     self.model.update_params(Q = Q_new)
#                     self.Q = self.model.get_param('Q')

#                 #   # if counter % 5 == 0:
#                 # self.q = self.conformal_p_control.compute_quantile()

#                 # # # Update the model's Q matrix based on the calibrated quantile
#                 # self.model.update_params(Q = np.pow(self.q, 2) * self.Q)
#                 # self.Q = self.model.get_param('Q')
                
#                 print(f"Updated Q : {self.Q}, q: {self.q}")  
#                 self.model.update(obs)
                
#         return self.model
                
#     def get_calibrated_model(self) -> LinearModel_Estimator:
#         return self.model
            