import numpy as np
from models.linear_model import LinearModel_Estimator
import copy
from collections import deque

class NLLThresholdCalibrator:
    """
    A calibrator that updates Q directly using a binary error based on NLL exceeding a threshold,
    designed for comparison with BCUC_Calibrator and DirectQCalibrator.
    
    Parameters:
    - initial_model: Initial linear model estimator.
    - gamma: Smoothing factor for Q updates (0 < gamma <= 1).
    - k: Window size for innovations and errors.
    - alpha: Target error rate (used for Q adjustment).
    - threshold_nll: NLL threshold for binary error (e.g., 1.0 for Gaussian baseline).
    - lambda_: Smoothing factor for Q update (0 < lambda_ <= 1).
    - update_after_every: Update Q after every n observations.
    - Q_clip: Tuple (min_Q, max_Q) for clipping Q.
    """
    def __init__(
        self,
        initial_model: LinearModel_Estimator,
        gamma=0.5,
        k=50,
        alpha=0.32,
        threshold_nll=1.0,  # Baseline NLL threshold (e.g., 0.5 * log(2π) + 0.5 for Gaussian)
        lambda_=0.01,
        update_after_every=15,
        Q_clip=(1e-8, 1e6)
    ):
        self.model = copy.deepcopy(initial_model)
        self.gamma = gamma
        self.k = k
        self.alpha = alpha
        self.threshold_nll = threshold_nll
        self.lambda_ = lambda_
        self.update_after_every = update_after_every
        self.Q_clip = Q_clip
        
        self.initial_Q = self.model.get_param('Q')  # Initial Q matrix
        self.n = self.model.get_dimention()  # Model dimension
        self.Q = self.initial_Q
        
        # Windows for errors and innovations
        self.E = deque(maxlen=k)  # Binary errors based on NLL
        self.innovations = deque(maxlen=k)  # Squared residuals for variance estimate
    
    def _proj_Q(self, Q):
        """Project Q to ensure it stays within bounds."""
        return np.clip(Q, self.Q_clip[0], self.Q_clip[1])

    def _update_Q_direct(self, E_bar):
        """Update Q using the mean of innovations, adjusted by NLL-based error rate."""
        if self.innovations:
            Q_innov = np.mean(self.innovations)
            # Adjust Q based on error rate deviation from alpha
            eta = min(0.3, 0.1 * max(1e-6, np.mean([abs(i) for i in self.innovations])))  # Simple eta based on innovation scale
            Q_adjust = eta * (E_bar - self.alpha)
            Q_new = (1 - self.lambda_) * self.Q + self.lambda_ * (Q_innov + Q_adjust)
            return self._proj_Q(Q_new)
        return self.Q

    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        # Reset run logs and model
        self.interval_widths = []
        self.Q_history = []
        self.nll_history = []
        self.meu_history = []
        self.Q = self.initial_Q
        if reset_obs:
            self.model.reset()

        # Shape checks
        if self.n == 1:
            n_samples, n_observations = observations.shape
            n_features = 1
        else:
            n_samples, n_observations, n_features = observations.shape

        if n_features != self.Q.shape[0]:
            raise ValueError("Number of features in observations must match the model's Q matrix dimensions.")

        if actions is None:
            actions = np.zeros((n_samples, n_observations, n_features))

        counter = 0
        for obs, action in zip(observations, actions):
            self.Q_history.append(self.Q.flatten().copy())
            meu, cov = self.model.predict(action)
            self.meu_history.append(meu.flatten().copy())

            # Compute standard deviations for metrics
            if self.model.state_dim == 1:
                std_vec = np.array([float(np.sqrt(cov))])
                var_vec = np.array([float(cov)])
            else:
                if cov.ndim == 2:
                    var_vec = np.diag(cov)
                else:
                    var_vec = np.asarray(cov).reshape(-1)
                var_vec = np.clip(var_vec, 1e-12, None)
                std_vec = np.sqrt(var_vec)

            self.interval_widths.append(2.0 * std_vec.copy())

            # NLL and error if observation available
            if not np.isnan(obs).any():
                y_vec = np.asarray(obs).reshape(-1)
                mu_vec = np.asarray(meu).reshape(-1)
                resid2 = (y_vec - mu_vec) ** 2
                nll_dims = 0.5 * (np.log(2 * np.pi * var_vec) + resid2 / np.clip(var_vec, 1e-12, None))
                self.nll_history.append(float(np.sum(nll_dims)))

                # Binary error based on NLL threshold
                e_t = 1 if np.sum(nll_dims) > self.threshold_nll else 0
                self.E.append(e_t)
                self.innovations.append(resid2)

                counter += 1
                if counter % self.update_after_every == 0:
                    E_bar = np.mean(self.E) if self.E else 0
                    Q_new = self._update_Q_direct(E_bar)
                    self.model.update_params(Q=Q_new)
                    self.Q = self.model.get_param('Q')

                self.model.update(obs)

        return self.model

    def get_calibrated_model(self) -> LinearModel_Estimator:
        return self.model

    def get_metrics(self, targetQ: np.ndarray = 0.5) -> dict:
        """
        Returns metrics: E_bar (mean NLL-based error), coverage (not directly controlled),
        mean interval width, avg NLL, and final difference from target Q.
        """
        E_bar = float(np.mean(self.E)) if self.E else np.nan
        coverage = 1.0 - E_bar if self.E else np.nan

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