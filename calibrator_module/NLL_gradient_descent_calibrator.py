import numpy as np
from collections import deque
import copy

class NLLBasedQCalibrator:
    def __init__(self,
                 initial_model,
                 learning_rate: float = 1e-2,
                 update_after_every: int = 15,
                 Q_clip: tuple = (1e-8, 1e6)):
        self.model = copy.deepcopy(initial_model)
        self.learning_rate = learning_rate
        self.update_after_every = update_after_every
        self.Q_min, self.Q_max = Q_clip

        self.initial_Q = self.model.get_param('Q')  
        self.n = self.model.get_dimention()
        self.Q = self.initial_Q

        self.interval_widths = []
        self.Q_history = []
        self.nll_history = []
        self.meu_history = []
        self.coverage_flags = []

        self._batch_grad_u = 0.0  # accumulate grad wrt log Q

    def _proj_Q(self, q_scalar: float) -> float:
        return float(np.clip(q_scalar, self.Q_min, self.Q_max))

    def calibrate_model(self, observations: np.ndarray, actions: np.ndarray = None, reset_obs=True):
        # reset logs
        self.interval_widths.clear()
        self.Q_history.clear()
        self.nll_history.clear()
        self.meu_history.clear()
        self.coverage_flags.clear()
        self._batch_grad_u = 0.0
        self.Q = self.initial_Q

        if reset_obs:
            self.model.reset()

        if self.n != 1:
            raise ValueError("This calibrator is designed for 1D only.")

        n_samples = observations.shape[0] if observations.ndim == 1 else observations.shape[0]
        if actions is None:
            actions = np.zeros((n_samples, 1, 1))

        # try to fetch R (measurement noise) if available
        try:
            R = float(self.model.get_param('R')[0, 0])
        except Exception:
            R = 0.0

        counter = 0
        for obs, action in zip(observations, actions):
            # log current Q (as scalar)
            self.Q_history.append(self.Q)

            # predict
            mu, cov = self.model.predict(action)  # cov here should be innovation variance S_t in 1D
            S = float(cov[0, 0])                 # innovation variance (predicted y variance)
            self.meu_history.append(mu)
            sigma = np.sqrt(max(S, 1e-12))
            self.interval_widths.append(2.0 * sigma)

            if not np.isnan(obs).any():
                y = float(np.asarray(obs).reshape(-1)[0])
                mu_s = float(np.asarray(mu).reshape(-1)[0])
                nu = y - mu_s                        # innovation
                # innovation-NLL (correct for process-noise learning)
                nll = 0.5 * (np.log(2*np.pi) + np.log(max(S, 1e-12)) + (nu*nu)/max(S, 1e-12))
                self.nll_history.append(float(nll))

                # nominal 68% interval with sigma -> coverage flag
                in_int = (mu_s - sigma <= y <= mu_s + sigma)
                self.coverage_flags.append(in_int)

                # gradient wrt log Q (approx): (1 - nu^2/S)/2 scaled by sensitivity psi_t
                # psi_t ≈ fraction of S due to state uncertainty (depends on Q)
                psi = (S - R) / max(S, 1e-12)
                psi = float(np.clip(psi, 0.0, 1.0))
                grad_u = 0.5 * (1.0 - (nu*nu)/max(S, 1e-12)) * psi
                self._batch_grad_u += grad_u

                counter += 1
                if counter % self.update_after_every == 0:
                    # log-Q update
                    u = float(np.log(self.Q))
                    u -= self.learning_rate * (self._batch_grad_u / self.update_after_every)
                    Q_new = self._proj_Q(np.exp(u))
                    # commit
                    self.model.update_params(Q=np.array([[Q_new]], dtype=float))
                    self.Q = float(self.model.get_param('Q')[0, 0])
                    self._batch_grad_u = 0.0

                # standard KF update with obs
                self.model.update(obs)

        return self.model

    def get_calibrated_model(self):
        return self.model

    def get_metrics(self, targetQ=0.5):
        E_bar = float(1 - np.mean(self.coverage_flags)) if self.coverage_flags else np.nan
        coverage = 1.0 - E_bar if not np.isnan(E_bar) else np.nan
        mean_width = float(np.mean(self.interval_widths)) if self.interval_widths else np.nan
        avg_nll = float(np.mean(self.nll_history)) if self.nll_history else np.nan
        return {
            "E_bar": E_bar,
            "coverage": coverage,
            "mean_interval_width": mean_width,
            "avg_nll": avg_nll,
            "final_difference": float(abs(self.Q - targetQ)),
            "relative_error": float(abs(self.Q - targetQ) / targetQ)
        }
