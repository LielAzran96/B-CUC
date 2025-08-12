import numpy as np
import pandas as pd
from typing import Optional, Iterable, Union

def print_for_debug(to_print : str, flag : bool = False):
    if flag:
        print(to_print)
                    
class ConformalPcontrol:
    """
    Conformal-P controller for online coverage calibration.

    This class maintains running lists of:
      - S: nonconformity/scale scores (e.g., s_t = |y - μ| / σ in 1D, or Mahalanobis in nD)
      - E: binary coverage errors (1 = miss, 0 = covered)
      - C: the current prediction interval (pandas.Interval; 1D only)
      - q: a scalar-like controller state used by the caller (can represent a quantile or any 1D control)
      - eta: current step size (computed from S via beta and eta_max)

    IMPORTANT SHAPE/USAGE NOTES (aligned with your B-CUC & NLL code):
      * 1D case (n == 1):
          - `meu`, `obs`, `sigma` are treated as scalars or shape (1,) arrays.
          - `compute_interval` builds a single pandas.Interval using μ ± q·σ.
          - `compute_error` checks membership of the scalar obs in that Interval.

      * nD case (n > 1):
          - `compute_score` supports sigma as a covariance matrix and uses Mahalanobis distance.
          - `compute_interval` is STILL 1D-ONLY by design here (uses pandas.Interval),
            so callers typically pass a scalar std (or the 1D projection they care about).
            We do NOT change this logic.

      * No logic has been changed compared to your original implementation.
        This file only adds documentation, explicit shapes, and clarifying comments.
    """

    def __init__(
        self,
        q_0: Union[float, np.ndarray],
        alpha: float,
        beta: float,
        n: int,
        eta_max: float,
        gamma: float,
        k: int = 75
    ):
        """
        Parameters
        ----------
        q_0 : float or np.ndarray
            Initial controller state (often a quantile). In your flows it is scalar-like (e.g., np.array([2.0])).
        alpha : float
            Target error rate (1 - target coverage). E.g., 0.32 for μ±σ under Gaussian residuals.
        beta : float
            Step-size base used inside `compute_eta` (eta = min(eta_max, beta * max(S))).
        n : int
            Model/state dimension. Used to switch between 1D vs nD score calculations.
        eta_max : float
            Upper bound for the step size eta.
        gamma : float
            Smoothing/mixing parameter (not used in current active logic—kept for parity with your code).
        k : int
            Number of last scores and errors to keep in S and E (default 50).
        """
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.q = q_0
        self.k = k #keeping the last k values in E and S
        # Running logs
        self.S: list = []          # scores s_t
        self.E: list = []          # errors e_t ∈ {0,1}
        self.E_bar: list = []      # history of mean(E) snapshots
        self.C: Optional[pd.Interval] = None  # last 1D interval

        # Step-size & dimension
        self.eta: Optional[float] = None
        self.n = n
        self.eta_max = eta_max

    def compute_interval(
        self,
        meu: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray]
    ) -> Optional[pd.Interval]:
        """
        Build a 1D prediction interval around `meu` using the current q and `sigma`.

        NOTE: This is intentionally 1D-only (uses pandas.Interval) and matches your original logic.
              For nD, callers usually pass a scalar std (e.g., per-dim) they care about.

        Parameters
        ----------
        meu : float or array-like (broadcastable to scalar via `.item()`)
            Predicted mean at current step (1D assumed for interval creation).
        sigma : float or array-like (broadcastable to scalar via `.item()`)
            Scale (std) for 1D interval; if you pass covariance in nD flows,
            you should first reduce it to a scalar std yourself.

        Returns
        -------
        pd.Interval
            Closed interval [μ - q·σ - margin, μ + q·σ + margin].

        Raises
        ------
        ValueError
            If the computed lower bound exceeds the upper bound.
        """
        margin = 1e-3  # small margin to avoid boundary false-positives
        lower_bound = (meu - self.q * sigma - margin).item()
        upper_bound = (meu + self.q * sigma + margin).item()

        if lower_bound > upper_bound:
            print("Warning: Lower bound is greater than upper bound. Adjusting bounds.")
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            print(f"Meu: {meu}, Sigma: {sigma}, q: {self.q}")
            raise ValueError("Lower bound cannot be greater than upper bound.")

        try:
            # Closed on both sides: obs on boundary counts as covered.
            self.C = pd.Interval(left=lower_bound, right=upper_bound, closed='both')
        except Exception as e:
            print(f"Error occurred while creating interval: {e}")
            print(f"Meu: {meu}, Sigma: {sigma}, q: {self.q}")
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            raise e

        return self.C

    def compute_error(self, obs: Union[float, np.ndarray]) -> int:
        """
        Compute binary coverage error e_t for a 1D observation against the last interval C.

        Parameters
        ----------
        obs : float or array-like (broadcastable to scalar)
            Actual observation at current step.

        Returns
        -------
        int
            e_t ∈ {0,1}, where 1 indicates a miss (not inside interval), 0 indicates coverage.

        Notes
        -----
        - Matches your original logic: `e_t = 1 - int(obs in self.C)`.
        - This assumes `self.C` is a pandas.Interval (1D).
        """
        e_t = 1 - int(obs in self.C)
        self.E.append(e_t)
  
        print_for_debug(f"e_t:{e_t}, mean_error = {np.mean(self.E)}")
        return e_t

    def compute_E_bar(self) -> float:
        """
        Return the running mean of coverage errors.

        Returns
        -------
        float
            mean(E) if E is non-empty, else 0.0 (exactly as your original).
        """
        if len(self.E):
            if len(self.E) > self.k:
                return float(np.mean((self.E[-self.k:])))
            else:
                return float(np.mean(self.E))
        else:
            return 0.0

    def compute_score(
        self,
        obs: Union[float, np.ndarray],
        meu: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute and log the nonconformity/scale score s_t.

        Parameters
        ----------
        obs : float or np.ndarray
            Observation y_t. For n==1, scalar-like; for n>1, typically a column vector (d×1) or 1D array (d,).
        meu : float or np.ndarray
            Predicted mean μ_t with the same shape convention as obs.
        sigma : float or np.ndarray
            For n==1: standard deviation (scalar-like).
            For n>1: covariance matrix (d×d). (Matches your original nD branch.)

        Returns
        -------
        float or np.ndarray
            s_t. In 1D, |y - μ| / σ (scalar).
            In nD, sqrt( (y-μ)^T Σ^{-1} (y-μ) ) (scalar-like).

        Notes
        -----
        - Logic unchanged. For n>1 we assume `sigma` is a covariance matrix.
        """
        if self.n == 1:
            s_t = np.abs(obs - meu) / sigma
        else:
            s_t = np.sqrt((obs - meu).T @ np.linalg.inv(sigma) @ (obs - meu))
        self.S.append(s_t)
        if len(self.S) > self.k:
            self.S = self.S[-self.k:]
        return s_t

    def compute_S_max(self) -> float:
        """
        Return the current maximum of recorded scores S.

        Returns
        -------
        float
            max(S) if S is non-empty, else 0.0 (exactly as your original).
        """
        
        return float(np.max(self.S)) if len(self.S) else 0.0

    def compute_eta(self) -> Optional[float]:
        """
        Compute step size η_t based on recorded scores.

        Returns
        -------
        float
            eta = min(eta_max, beta * max(S)), or 0 if S is empty. Also stored in `self.eta`.

        Notes
        -----
        - Prints the computed eta and the current max score, matching your original behavior.
        - This function only computes and stores eta; it does not update q or Q.
        """
        self.eta = min(self.beta * np.max(self.S) if self.S else 0, self.eta_max)
        print_for_debug(f"Computed eta: {self.eta}, eta_max: {self.eta_max}")
        print_for_debug(f"Max score: {np.max(self.S) if self.S else 0}")
        return self.eta

    def compute_quantile(self, Q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Update and return the controller state using coverage error (original logic kept verbatim).

        Parameters
        ----------
        Q : float or np.ndarray
            External scalar-like state passed in from the caller (your original code names this argument `Q`).

        Returns
        -------
        float or np.ndarray
            Updated state (named Q_t1 here, but assigned back into `self.q` exactly like your original).

        Notes
        -----
        - This method keeps your original math and naming (Q used as input; output assigned to self.q).
        - The lower bound floor is kept: max(Q_t1, 0.0001).
        - We append the current mean(E) to E_bar history (unchanged).
        """
        total_mean = self.compute_E_bar()
        self.E_bar.append(total_mean)
        mean = total_mean  # original choice (sometimes you tried last-k; here it's all-time mean)
        name = "mean_last_k"
        delta = mean - self.alpha
        print_for_debug(f"{name}: {mean}")

        # === original update logic preserved ===
        Q_t1 = Q + self.eta * delta
        Q_t1 = max(Q_t1, np.array([0.0001]))  # keep a positive floor (unchanged)

        # original line: assign back into self.q (naming retained)
        self.q = Q_t1
        return Q_t1
