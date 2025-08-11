import numpy as np
import copy

class NLLGD_Calibrator:
    """
    Online NLL-based calibrator for process noise Q (no quantile).
    Evaluation uses μ ± σ for coverage and width — same keys as BCUC.get_metrics().
    """

    def __init__(
        self,
        initial_model,
        eta_Q: float = 1e-2,     # learning rate for Q
        kappa: float = 1.0,      # d(sigma^2)/dQ sensitivity (≈1.0 for many linear KFs)
        Q_min: float = 1e-8,
        Q_max: float = 1e6,
        clip_grad: float = None, # optional gradient clipping
        ema_beta: float = None,  # optional EMA for gradient smoothing (e.g., 0.9)
        reset_model_on_start: bool = True,
    ):
        self.model = copy.deepcopy(initial_model)
        self.eta_Q = float(eta_Q)
        self.kappa = float(kappa)
        self.Q_min = float(Q_min)
        self.Q_max = float(Q_max)
        self.clip_grad = clip_grad
        self.ema_beta = ema_beta
        self.reset_model_on_start = reset_model_on_start

        # state
        self.Q = self.model.get_param('Q')
        self.n = self.model.get_dimention()
        self._g_ema = None

        # logs (aligned with BCUC)
        self.interval_widths = []  # scalar per step: sum(2*sigma)
        self.nll_history = []      # per-step NLL (sum over dims), BEFORE updates
        self.coverage_hits = []    # 1 if y in [mu ± sigma] (all dims), else 0
        self.Q_history = []
        self.mu_history = []

    # ---------- internals ----------
    @staticmethod
    def _ensure_diag(Q, n):
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
        diag_q = np.asarray(diag_q).reshape(-1)
        if diag_q.size == 1:
            return np.array([[float(diag_q[0])]])
        return np.diag(diag_q)

    @staticmethod
    def _safe_var_and_std(cov):
        cov = np.asarray(cov)
        if cov.ndim == 0:
            var = float(cov)
            return np.array([var]), np.array([float(np.sqrt(max(var, 1e-12)))])
        if cov.ndim == 1:
            var = cov
            return var, np.sqrt(np.clip(var, 1e-12, None))
        if cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            if cov.shape[0] == 1:
                var = float(cov[0, 0])
                return np.array([var]), np.array([float(np.sqrt(max(var, 1e-12)))])
            var = np.diag(cov)
            return var, np.sqrt(np.clip(var, 1e-12, None))
        raise ValueError("Unexpected covariance shape in _safe_var_and_std.")

    @staticmethod
    def _nll_gaussian(y, mu, var_vec):
        y = np.asarray(y).reshape(-1)
        mu = np.asarray(mu).reshape(-1)
        var = np.asarray(var_vec).reshape(-1)
        var = np.clip(var, 1e-12, None)
        resid2 = (y - mu) ** 2
        nll_dims = 0.5 * (np.log(2 * np.pi * var) + resid2 / var)
        return float(np.sum(nll_dims)), resid2

    def _grad_wrt_Q_diag(self, var_vec, resid2):
        var = np.asarray(var_vec).reshape(-1)
        inv = 1.0 / np.clip(var, 1e-12, None)
        # dNLL/dσ2 = 0.5*(1/σ2) - 0.5*resid2/σ4
        grad_sigma2 = 0.5 * inv - 0.5 * resid2 * (inv ** 2)
        grad_Q = self.kappa * grad_sigma2         # dσ2/dQ ≈ kappa
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

    def _proj_Q(self, Q):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 0:
            return float(np.clip(Q, self.Q_min, self.Q_max))
        if Q.ndim == 1:
            return np.clip(Q, self.Q_min, self.Q_max)
        return np.clip(Q, self.Q_min, self.Q_max)

    def _as_control_vec(self, action):
        """Coerce action to a 1-D control vector matching B.shape[1]."""
        u = np.asarray(action).reshape(-1)
        try:
            u_dim = int(self.model.B.shape[1])
        except Exception:
            u_dim = u.size
        if u.size == u_dim:
            return u
        if u.size == 1 and u_dim > 1:
            return np.full((u_dim,), float(u[0]))
        raise ValueError(f"Action length {u.size} != B.shape[1] {u_dim}")
    # --------------------------------

    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        # reset logs
        self.interval_widths = []
        self.nll_history = []
        self.coverage_hits = []
        self.Q_history = []
        self.mu_history = []

        # shape handling identical to B-CUC
        if self.n == 1:
            n_samples, n_observations = observations.shape
            n_features = 1
        else:
            n_samples, n_observations, n_features = observations.shape

        # default actions just like B-CUC
        if actions is None:
            actions = np.zeros((n_samples, n_observations, n_features))
        if self.reset_model_on_start and reset_obs:
            self.model.reset()

        q_diag = self._ensure_diag(self.Q, self.n)

        # iterate exactly like B-CUC
        for obs, action in zip(observations, actions):
            self.Q_history.append(np.copy(q_diag))

            # coerce control to 1-D vector
            a_vec = self._as_control_vec(action)

            # predict
            mu_t, cov_t = self.model.predict(a_vec)
            self.mu_history.append(np.asarray(mu_t).flatten().copy())

            var_vec, std_vec = self._safe_var_and_std(cov_t)
            if self.model.state_dim == 1:
                std_vec = np.array([float(np.sqrt(cov_t))])  # 1D -> vector of length 1
                var_vec = np.array([float(cov_t)])
            else:
                if cov_t.ndim == 2:
                    var_vec = np.diag(cov_t)
                else:
                    var_vec = np.asarray(cov_t).reshape(-1)
                var_vec = np.clip(var_vec, 1e-12, None)
                std_vec = np.sqrt(var_vec)
                
            self.interval_widths.append(float(np.sum(2.0 * std_vec)))

            # update if observation available
            if not np.isnan(obs).any():
                
                y_vec = np.asarray(obs).reshape(-1)
                mu_vec = np.asarray(mu_t).reshape(-1)
                if len(self.nll_history) < 20:
                    print(f"resid2/var mean: {float(np.mean((y_vec - mu_vec)**2 / np.clip(var_vec,1e-12,None))):.3g}, "
                        f"var mean: {float(np.mean(var_vec)):.3g}")
                # coverage with μ ± σ
                lower = mu_vec - std_vec
                upper = mu_vec + std_vec
                hit = int(np.all((y_vec >= lower) & (y_vec <= upper)))
                self.coverage_hits.append(hit)

                # NLL (before update)
                nll_t, resid2 = self._nll_gaussian(y_vec, mu_vec, var_vec)
                self.nll_history.append(nll_t)

                # ---- NLL gradient step on diag(Q) ----
                grad_q = self._grad_wrt_Q_diag(var_vec, resid2)
                grad_q = self._apply_ema_and_clip(grad_q)
                q_diag = q_diag - self.eta_Q * grad_q
                q_diag = self._proj_Q(q_diag)

                Q_new = self._from_diag(q_diag)
                self.model.update_params(Q=Q_new)
                self.Q = self.model.get_param('Q')

                # assimilate observation
                self.model.update(obs)

        return self.model

    def get_calibrated_model(self):
        return self.model

    def get_metrics(self):
        """
        Matches BCUC keys:
          - E_bar: mean(1 - hit) using μ ± σ
          - coverage: 1 - E_bar
          - mean_interval_width: avg sum(2σ)
          - avg_nll: mean NLL
        """
        if self.coverage_hits:
            E_bar = float(np.mean([1 - h for h in self.coverage_hits]))
            coverage = 1.0 - E_bar
        else:
            E_bar = np.nan
            coverage = np.nan
        mean_width = float(np.mean(self.interval_widths)) if self.interval_widths else np.nan
        avg_nll = float(np.mean(self.nll_history)) if self.nll_history else np.nan
        return {
            "E_bar": E_bar,
            "coverage": coverage,
            "mean_interval_width": mean_width,
            "avg_nll": avg_nll,
        }
