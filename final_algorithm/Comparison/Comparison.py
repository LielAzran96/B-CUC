import sys
import os
sys.path.append(os.path.abspath("../../"))  # or "." or ".." depending on your structure
import numpy as np
import matplotlib.pyplot as plt
from models.model_utils import mask_observations, params
from models.linear_model import LinearModel_Estimator
from calibrator_module.BCUC_calibrator_finalVersion import BCUC_Calibrator
from models.model_utils import params
from calibrator_module.NLL_gradient_descent_calibrator import NLLBasedQCalibrator
from calibrator_module.conformal_prediction_classic import conformalPrediction_Calibrator
import pandas as pd
from utils import plot_Qt, _to_1d_float
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run q_Q explosion experiment.")
    parser.add_argument("--initial_Q", type=float, default="0.6275", help="initial Q to start the calibration")
    return parser.parse_args()

def main(initial_Q):
    # --- Load data ---
    params['Q'] = np.array([initial_Q])  # Set initial Q value
    persons = [0.01, 0.3, 0.7, 1.5]
    initial_model = LinearModel_Estimator(**params)
    print(f"Initial Q matrix: {initial_model.Q}")

    calibrators_BCUC = []
    calibrators_conformalPred = []
    calibrators_nll = []

    calibrated_models_BCUC = []
    calibrated_models_conformalPred = []
    calibrated_models_nll = []

    metrics_BCUC = []
    metrics_conformalPred = []
    metrics_nll = []

    # --
    for person in persons:
        path_to_load = f"../../observations/observations_random_zeros_and_actions_for_Q{person}_steps_10000.npz"
        data = np.load(path_to_load, allow_pickle=True)
        masked_observations = mask_observations(data['observations'])
        actions = data['actions']

        # --- build models ---
        to_print = f"Target (true) Q*: {person}"
        print(to_print)
        
        cal_qc = BCUC_Calibrator(
            initial_model=initial_model,
            alpha=0.32,          # μ±σ target; keeps Q identification unbiased if Gaussian
        )
        model_qc = cal_qc.calibrate_model(masked_observations, actions)
        calibrated_models_BCUC.append(model_qc)
        calibrators_BCUC.append(cal_qc)

        cal_cp = conformalPrediction_Calibrator(
        initial_model=initial_model,
        update_after_every=15
        )
        
        model_cp = cal_cp.calibrate_model(masked_observations, actions)
        calibrated_models_conformalPred.append(model_cp)
        calibrators_conformalPred.append(cal_cp)

        cal_nll = NLLBasedQCalibrator(
        initial_model=initial_model,
        update_after_every=15
        )
        
        model_nll = cal_nll.calibrate_model(masked_observations, actions)
        calibrated_models_nll.append(model_nll)
        calibrators_nll.append(cal_nll)
        
        NLLBasedQCalibrator
        # --- evaluate metrics ---
        metrics_qc = cal_qc.get_metrics(person)
        metrics_BCUC.append(metrics_qc)
        metrics_cp = cal_cp.get_metrics(person)
        metrics_conformalPred.append(metrics_cp)
        metrics_nll_dict = cal_nll.get_metrics(person)
        metrics_nll.append(metrics_nll_dict)
        
        # --- print results ---
        print("[BCUC quantile_coupled]")
        print("metrics:", metrics_qc)
        print("final Q:", model_qc.Q)
        print()
        print("[Conformal Prediction quantile_coupled Threshold Calibrator]")
        print("metrics:", metrics_cp)
        print("final Q:", model_cp.Q)
        print()
        print("[NLL Based Q Calibrator]")
        print("metrics:", metrics_nll_dict)
        print("final Q:", model_nll.Q)
        print()
        
    metrics_BCUC = pd.DataFrame(metrics_BCUC, index=persons)
    metrics_conformalPred = pd.DataFrame(metrics_conformalPred, index=persons)
    metrics_nll = pd.DataFrame(metrics_nll, index=persons)


    data = {
        "BCUC": [
            metrics_BCUC["E_bar"].mean(),
            metrics_BCUC["mean_interval_width"].mean(),
            metrics_BCUC["final_difference"].mean(),
            metrics_BCUC["avg_nll"].mean()
        ],
        "Conformal": [
            metrics_conformalPred["E_bar"].mean(),
            metrics_conformalPred["mean_interval_width"].mean(),
            metrics_conformalPred["final_difference"].mean(),
            metrics_conformalPred["avg_nll"].mean()
        ],
        "NLL": [
            metrics_nll["E_bar"].mean(),
            metrics_nll["mean_interval_width"].mean(),
            metrics_nll["final_difference"].mean(),
            metrics_nll["avg_nll"].mean()
        ]
    }

    df_summary = pd.DataFrame(
        data,
        index=["E_bar_mean", "mean_interval_width_mean", "final_difference_mean", "avg_nll_mean"]
    )

    print(df_summary)

    Q_vals_BCUC = [cal.Q_history for cal in calibrators_BCUC]
    Q_vals_conformal= [cal.Q_history for cal in calibrators_conformalPred]
    Q_vals_nll= [cal.Q_history for cal in calibrators_nll]

    q_vals_BCUC = [cal.q_history for cal in calibrators_BCUC]
    q_vals_conformal = [cal.q_history for cal in calibrators_conformalPred]
    
    dir_name = "plots/all_together"
    os.makedirs(dir_name, exist_ok=True)    
    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Time Step")
    ax.set_title("Q_t")
    ax.grid(True)

    for i, (Q_val, Q_val_conformal, Q_val_nll) in enumerate(zip(Q_vals_BCUC, Q_vals_conformal, Q_vals_nll)):
        y  = _to_1d_float(Q_val)
        y2 = _to_1d_float(Q_val_conformal)
        y3 = _to_1d_float(Q_val_nll)

        x  = np.arange(len(y))
        line, = ax.plot(x, y, linewidth=1.5,
                        label=f"Q_t - BCUC quantile_coupled (Person {persons[i]})",
                        linestyle='--')
        color = line.get_color()

        x2 = np.arange(len(y2))
        ax.plot(x2, y2, linewidth=1.5,
                label=f"Q_t - conformal prediction (Person {persons[i]})",
                color=color, linestyle='-.')

        x3 = np.arange(len(y3))
        ax.plot(x3, y3, linewidth=1.5,
                label=f"Q_t - NLL - direct Q update (Person {persons[i]})",
                color=color, linestyle=':')

        ax.axhline(y=float(np.asarray(persons[i]).reshape(-1)[0]),
                color=color, linestyle='-',
                label=f"Target Q* = {persons[i]}")

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "Q_t_over_time.png"), dpi=300, bbox_inches='tight')
    print("Q_t plot saved at:", os.path.join(dir_name, "Q_t_over_time.png"))
    
    #a different plot for Q_t if needed 
    # Call the plotting function
    plot_Qt(
        Q_vals_BCUC,        # your BCUC q_t values
        Q_vals_conformal,   # your conformal q_t values
        Q_vals_nll,         # your NLL q_t values
        persons,   # your Q* targets (the same as in your legend)
        dir_name=dir_name,
        title="Q_t over time"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Time Step")
    ax.set_title("q_t")
    ax.grid(True)


    for i, (q_val, q_val_conformal) in enumerate(zip(q_vals_BCUC, q_vals_conformal)):
        y  = _to_1d_float(q_val)
        y2 = _to_1d_float(q_val_conformal)

        # If lengths differ, align to the shorter to avoid shape errors
        T = min(len(y), len(y2))
        x = np.arange(T)

        line, = ax.plot(x, y[:T], linewidth=2,
                        label=f"q_t - BCUC quantile_coupled (Person {persons[i]})")
        color = line.get_color()

        ax.plot(x, y2[:T], linewidth=2,
                label=f"q_t - conformal prediction (Person {persons[i]})",
                color=color, linestyle='--')

    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "q_t_over_time_.png"), dpi=300, bbox_inches='tight')
    print("q_t plot saved at:", os.path.join(dir_name, "q_t_over_time_.png"))

    
if __name__ == "__main__":
    args = parse_args()
    print("Starting the analysis..")
    # === Run your pipeline ===
    main(initial_Q=args.initial_Q)