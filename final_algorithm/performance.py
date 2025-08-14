import argparse
import sys
import os
sys.path.append(os.path.abspath("../"))  # or "." or ".." depending on your structure
import numpy as np
import matplotlib.pyplot as plt
from models.model_utils import mask_observations, params
from models.linear_model import LinearModel_Estimator
from calibrator_module.BCUC_calibrator_finalVersion import BCUC_Calibrator
from models.model_utils import params

def parse_args():
    parser = argparse.ArgumentParser(description="Run q_Q explosion experiment.")
    parser.add_argument("--initial_Q", type=float, default="0.6275", help="initial Q to start the calibration")
    return parser.parse_args()

def main(initial_Q):
    
    # --- load data ---
    persons = [0.01, 0.3, 0.7, 1.5]
    params['Q'] = np.array([initial_Q])  # Set initial Q value
    initial_model = LinearModel_Estimator(**params)   # e.g., Q_initial = mean(persons)
    print(f"Initial Q matrix: {initial_model.Q}")
    calibrators = []
    calibrated_models = []
    metrics = []
    # --- 1) B-CUC: quantile-coupled (q_t -> anchored Q) ---
    cal_qc = BCUC_Calibrator(
        initial_model=initial_model,
        alpha=0.32,          # μ±σ target; keeps Q identification unbiased if Gaussian
    )
    for person in persons:
        path_to_load = f"../observations/observations_random_zeros_and_actions_for_Q{person}_steps_10000.npz"
        data = np.load(path_to_load, allow_pickle=True)
        masked_observations = mask_observations(data['observations'])
        actions = data['actions']

        # --- build model ---
        to_print = f"Target (true) Q*: {person}"
        print(to_print)
        
        cal_qc = BCUC_Calibrator(
        initial_model=initial_model,
        alpha=0.32,          # μ±σ target; keeps Q identification unbiased if Gaussian
        )
        model_qc = cal_qc.calibrate_model(masked_observations, actions)
        calibrated_models.append(model_qc)
        calibrators.append(cal_qc)

        # --- evaluate metrics ---
        metrics_qc = cal_qc.get_metrics(person)
        metrics.append(metrics_qc)
        # --- print results ---
        print("[BCUC quantile_coupled]")
        print("metrics:", metrics_qc)
        print("final Q:", model_qc.Q)
        print()
        
    Q_vals_qc = [cal_qc.Q_history for cal_qc in calibrators]

    q_vals_qc = [cal_qc.q_history for cal_qc in calibrators]

    interval_widths_qc = [cal_qc.interval_widths for cal_qc in calibrators]

    E_qc = [cal_qc.conformal_p_control.E for cal_qc in calibrators]

    interval_widths_with_correction_qc = [[val*q for val, q in zip(interval_widths_qc_p, q_vals_qc_p)] for interval_widths_qc_p, q_vals_qc_p in zip(interval_widths_qc, q_vals_qc)]
    sigma_qc = [[std/2 for std in interval_widths_with_correction_qc_p] for interval_widths_with_correction_qc_p in interval_widths_with_correction_qc] #this is the vlaues of the std but calculated right after the prediction 


    meu_qc = [[dictionary['mean'].flatten() for dictionary in model_qc.dist_estimates] for model_qc in calibrated_models]
    std_qc = [[np.sqrt(dictionary['var'].flatten()) for dictionary in model_qc.dist_estimates] for model_qc in calibrated_models]
    meu_qc = [np.array(meu).squeeze() for meu in meu_qc]
    std_qc = [np.array(std).squeeze() for std in std_qc]

    dir_name = "plots/all_together"
    os.makedirs(dir_name, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Time Step")
    ax.set_title("Q_t")
    ax.grid(True)

    for i, Q_val in enumerate(Q_vals_qc):
        y = [q[0] for q in Q_val]
        x = np.arange(len(y))
        line, = ax.plot(x, y, linewidth=2,
                        label=f"Q_t - BCUC quantile_coupled (Person {persons[i]})")
        color = line.get_color()
        ax.axhline(y=persons[i], color=color, linestyle='--',
                label=f"Target Q* = {persons[i]}")

    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "Q_t_over_time.png"),
                dpi=300, bbox_inches='tight')
    print("Q_t plot saved at:", os.path.join(dir_name, "Q_t_over_time.png"))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("q_t")
    ax.grid(True)

    for i, q_vals in enumerate(q_vals_qc):
        y = np.array(q_vals).flatten()
        x = np.arange(len(y))
        ax.plot(x, y, linewidth=2, label=f"q_t (Person {persons[i]})")

    # Legend outside and smaller
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "q_t.png"), dpi=300, bbox_inches='tight')
    print("q_t plot saved at:", os.path.join(dir_name, "q_t.png"))

    
if __name__ == "__main__":
    args = parse_args()
    print("Starting the analysis..")
    # === Run your pipeline ===
    main(initial_Q=args.initial_Q)
