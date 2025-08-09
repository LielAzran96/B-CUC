import numpy as np
import copy

class NLLGD_Calibrator:
    """
    Online NLL-based calibrator for process noise Q.
    At each observed step t:
        1) Get predictive mean/cov: mu_t, cov_t = model.predict(action_t)
        2) Compute NLL(y_t | mu_t, cov_t)
        3) Take a gradient step on Q to minimize NLL
        4) Update model params with new Q
        5) Assimilate observation: model.update(y_t)

    Assumptions:
      - For 1D: predictive variance sigma^2 = cov (scalar), and d(sigma^2)/dQ ≈ kappa (default 1.0).
      - For nD: uses marginal variances (diag of cov) and updates only the diagonal of Q.
    """

    def __init__(
        self,
        initial_model,
        eta_Q: float = 1e-2,           # learning rate for Q
        kappa: float = 1.0,            # sensitivity d(sigma^2)/dQ (≈1.0 for many linear KF setups)
        Q_min: float = 1e-6,           # lower bound to keep Q positive
        clip_grad: float = None,       # optional gradient clipping (e.g., 1.0)
        ema_beta: float = None,        # optional EMA for gradients (0<beta<1), e.g., 0.9
        interval_alpha: float = 0.15,  # for tracking intervals (not used in the update)
        z_for_interval: float = None,  # if None, uses Gaussian z from alpha
        reset_model_on_start: bool = True,
    ):
        self.model = copy.deepcopy(initial_model)
        self.eta_Q = float(eta_Q)
        self.kappa = float(kappa)
        self.Q_min = float(Q_min)
        self.clip_grad = clip_grad
        self.ema_beta = ema_beta

        self.alpha = interval_alpha
        if z_for_interval is None:
            # central (1 - alpha) two-sided interval → z_{1 - alpha/2}
            from math import sqrt, log, pi
            # approximate inverse CDF via scipy-less hack (or set a constant like 1.96 for alpha=0.05)
            # Here: use a simple mapping for common alphas; fallback to 1.0.
            lookup = {0.05: 1.959964, 0.1: 1.644854, 0.15: 1.439531, 0.2: 1.281552}
            self.z = lookup.get(round(self.alpha, 2), 1.439531)
        else:
            self.z = float(z_for_interval)

        # Pull initial Q
        self.Q = self.model.get_param('Q')
        self.n = self.model.get_dimention()

        # Histories
        self.Q_history = []
        self.nll_history = []
        self.interval_widths = []
        self.mu_history = []

        # EMA buffers
        self._g_ema = None  # 1D or vector for diag(Q)

        self.reset_model_on_start = reset_model_on_start

    @staticmethod
    def _ensure_diag(Q, n):
        """Return a 1D view of diagonal if Q is diagonal/1D; otherwise raise for safety."""
        Q = np.asarray(Q)
        if Q.ndim == 0:
            return Q.reshape(1)
        if Q.ndim == 1 and Q.size == n:
            return Q
        if Q.ndim == 2 and Q.shape[0] == Q.shape[1] == n:
            return np.diag(Q)
        if n == 1 and Q.size == 1:
            return np.array([float(Q)])
        raise ValueError("NLLGD_Calibrator expects scalar or diagonal Q.")

    @staticmethod
    def _from_diag(diag_q):
        """Build a diagonal matrix from a 1D array, or scalar if length==1."""
        diag_q = np.asarray(diag_q).reshape(-1)
        if diag_q.size == 1:
            return np.array([[float(diag_q[0])]])
        return np.diag(diag_q)

    @staticmethod
    def _safe_var_and_sigma(cov):
        cov = np.asarray(cov)
        if cov.ndim == 0:
            var = float(cov)
            return var, float(np.sqrt(max(var, 1e-12)))
        if cov.ndim == 1:
            # treat as diagonal variances
            var = cov
            return var, np.sqrt(np.clip(var, 1e-12, None))
        if cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            if cov.shape[0] == 1:
                var = float(cov[0, 0])
                return var, float(np.sqrt(max(var, 1e-12)))
            var = np.diag(cov)
            return var, np.sqrt(np.clip(var, 1e-12, None))
        raise ValueError("Unexpected covariance shape in _safe_var_and_sigma.")

    @staticmethod
    def _nll_gaussian(y, mu, var):
        """
        Elementwise NLL for Gaussian(s):
          0.5*log(2πσ^2) + (y - μ)^2 / (2σ^2)
        Supports scalar or per-dimension arrays.
        Returns sum over dimensions.
        """
        y = np.asarray(y).reshape(-1)
        mu = np.asarray(mu).reshape(-1)
        var = np.asarray(var)
        if var.ndim == 0:
            var = np.array([float(var)])
        elif var.ndim == 2:
            var = np.diag(var)
        var = np.clip(var.reshape(-1), 1e-12, None)

        resid2 = (y - mu) ** 2
        nll_dims = 0.5 * (np.log(2 * np.pi * var) + resid2 / var)
        return float(np.sum(nll_dims)), resid2

    def _grad_wrt_Q_diag(self, y, mu, var, resid2):
        """
        Gradient dNLL/dQ (diag):
          dNLL/dσ2 = 0.5*(1/σ2) - 0.5*resid2/σ4
          dσ2/dQ ≈ kappa  (user-supplied sensitivity; default 1.0)
        So: dNLL/dQ ≈ kappa * (0.5/σ2 - 0.5*resid2/σ4)
        Returns a vector aligned with diag(Q).
        """
        var = np.asarray(var).reshape(-1)
        resid2 = np.asarray(resid2).reshape(-1)
        inv = 1.0 / np.clip(var, 1e-12, None)
        grad_sigma2 = 0.5 * inv - 0.5 * resid2 * (inv ** 2)
        grad_Q = self.kappa * grad_sigma2
        return grad_Q

    def _apply_ema_and_clip(self, g):
        if self.ema_beta is not None:
            if self._g_ema is None:
                self._g_ema = np.copy(g)
            else:
                self._g_ema = self.ema_beta * self._g_ema + (1 - self.ema_beta) * g
            g = self._g_ema
        if self.clip_grad is not None:
            g = np.clip(g, -self.clip_grad, self.clip_grad)
        return g

    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        """
        Run online calibration on a sequence of observations (optionally with actions).
        observations shape:
          - 1D case: (T, 1) or (T,)  (we treat as (T,1))
          - nD case: (T, n_observations, n_features) or (T, n_features)

        Returns:
            calibrated_model
        """
        # Normalize shapes similarly to your B-CUC:
        if self.n == 1:
            # Expect (T, 1) or (T,)
            obs_arr = np.asarray(observations)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(-1, 1)
            n_samples = obs_arr.shape[0]
            n_obs = 1
            n_feat = 1
        else:
            obs_arr = np.asarray(observations)
            if obs_arr.ndim == 2:
                # (T, n_features) → (T, 1, n_features)
                obs_arr = obs_arr.reshape(obs_arr.shape[0], 1, obs_arr.shape[1])
            n_samples, n_obs, n_feat = obs_arr.shape

        # Actions
        if actions is None:
            actions = np.zeros_like(obs_arr)
        if self.reset_model_on_start and reset_obs:
            self.model.reset()

        # Extract/prepare Q as diagonal vector
        q_diag = self._ensure_diag(self.Q, self.n)

        for t in range(n_samples):
            # For your code style, you iterate over (obs, action) pairs possibly batched per step.
            # We’ll loop over the 'n_obs' items (often 1).
            for j in range(n_obs):
                y_t = obs_arr[t, j]  # shape (n_feat,)
                a_t = actions[t, j]

                # Log current Q
                self.Q_history.append(q_diag.copy())

                # Predict
                mu_t, cov_t = self.model.predict(a_t)
                self.mu_history.append(np.asarray(mu_t).flatten().copy())

                # Interval width logging (for comparison)
                var_t, sigma_t = self._safe_var_and_sigma(cov_t)
                if np.isscalar(self.z):
                    # Store total width (sum across dims), just like your 2*sigma convention
                    w = 2 * self.z * (sigma_t if np.isscalar(sigma_t) else np.linalg.norm(sigma_t, 1))
                else:
                    w = 2 * np.sum(self.z * sigma_t)  # rare case of per-dim z
                self.interval_widths.append(w)

                # If observation is missing, skip update but still pass it through model.update?
                if np.isnan(y_t).any():
                    # You can decide to call update with NaN or skip; mirroring your B-CUC, skip the learning.
                    continue

                # Compute NLL and gradient wrt diag(Q)
                nll_t, resid2 = self._nll_gaussian(y_t, mu_t, var_t)
                self.nll_history.append(nll_t)

                # Gradient (diag)
                grad_q = self._grad_wrt_Q_diag(y_t, mu_t, var_t, resid2)
                grad_q = self._apply_ema_and_clip(grad_q)

                # Gradient descent step on diag(Q)
                q_diag = q_diag - self.eta_Q * grad_q

                # Enforce lower bound
                q_diag = np.maximum(q_diag, self.Q_min)

                # Write back into model as diagonal matrix (or scalar for 1D)
                Q_new = self._from_diag(q_diag)
                self.model.update_params(Q=Q_new)
                self.Q = self.model.get_param('Q')  # keep in sync

                # Assimilate the observation
                self.model.update(y_t)

        return self.model

    def get_calibrated_model(self):
        return self.model
