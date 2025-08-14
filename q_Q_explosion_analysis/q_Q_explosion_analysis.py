import sys
import os
sys.path.append(os.path.abspath("../"))  
import numpy as np
import matplotlib.pyplot as plt
from models.model_utils import mask_observations, params
from models.linear_model import LinearModel_Estimator
from calibrator_module.modules_that_didnt_workout.BCUC_calibrator import BCUC_Calibrator
from models.model_utils import params
import argparse

def main(file_path, initial_Q, person_number):

    #load 
    data = np.load(file_path)
    masked_observations = mask_observations(data['observations'])
    actions = data['actions']
    params['Q'] = np.array([initial_Q])
    patient_1_model = LinearModel_Estimator(**params) 
    print(f"Initial Q matrix: {patient_1_model.Q}")
    print(f"wanted Q matrix: 0.0001")
    calibrator = BCUC_Calibrator(patient_1_model)  # Note: Capital 'C' in Calibrator
    patient_1_calibrated = calibrator.calibrate_model(masked_observations, actions)

    print(f"calibrated Q :{patient_1_calibrated.Q}")
    
    Q_vals = calibrator.Q_history
    q_vals = calibrator.q_history
    interval_widths = calibrator.interval_widths
    E = calibrator.conformal_p_control.E
    interval_widths_with_correction = [val*q for val, q in zip(interval_widths, q_vals)]
    sigma = np.array([float(np.asarray(w).reshape(-1)[0]) for w in interval_widths], dtype=float) / 2.0
    #step (without the update step in case of observation available)
    #it is necessary to follow this value to evaluate our calibration mechanism (we can check the interval width and cover precision)
    meu = [dictionary['mean'].flatten() for dictionary in patient_1_calibrated.dist_estimates]
    std = [np.sqrt(dictionary['var'].flatten()) for dictionary in patient_1_calibrated.dist_estimates]
    meu = np.array(meu[1:]).squeeze()   
    std = np.array(std[1:]).squeeze()

    dir_name = f"plots/{person_number}" 
    os.makedirs(dir_name, exist_ok=True)    
    
    plt.figure(figsize=(10, 5))
    plt.plot(Q_vals, label="Q_t")
    plt.xlabel("Time Step")
    plt.ylabel("Value ")
    plt.title(" Q_t")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = "Q_t_over_time.png"
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print("Q_t plot saved at:", full_path)

    
    plt.figure(figsize=(10, 5))
    plt.plot(interval_widths_with_correction, label="Prediction Interval Width")
    plt.xlabel("Time Step")
    plt.ylabel("Width")
    plt.title("Prediction Interval Width Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_name = "plot_interval_width.png"
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print("Interval width plot saved at:", full_path)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(q_vals)), q_vals,  label="q_t")
    # plt.yscale("log")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("q_t")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = "q_t.png"
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print("q_t plot saved at:", full_path)
    
    plt.figure(figsize=(10, 5))
    time_steps = range(len(E))  # or your actual time step data
    plt.scatter(time_steps, E, label="coverage error")
    # plt.yscale("log")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Binary Coverage Errors Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = "E_t.png"
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300)
    print("E_t plot saved at:", full_path)

    plt.figure(figsize=(10, 5))
    ts = np.arange(len(meu)) 
    plt.plot(ts, meu, label=" meu")
    # plt.fill_between(ts, meu- std, meu + std, alpha=0.2, label="Confidence Interval")
    # plt.yscale("log")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Mean and Standard Deviation Over Time vs Observations")
    plt.grid(True)
    plt.tight_layout()
    plt.fill_between(ts, meu - sigma, meu + sigma, alpha=0.2, label="Confidence Interval")
    ts_new = np.arange(np.size(masked_observations))
    # mask = ~np.isnan(masked_observations)
    # ts_valid = ts_new[mask.flatten()]
    # obs_valid = masked_observations[mask.flatten()]
    # plt.scatter(ts_valid, obs_valid, color='red', label='Observations')
    obs = data['observations']
    plt.scatter(ts_new, obs, color='red', label='Observations')
    fig_name = "calibrated_distribution_over_time_vs_observation.png"
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300)
    print("Distribution plot saved at:", full_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Run q_Q explosion experiment.")
    parser.add_argument("--file_path", type=str, default="../observations/observations_random_zeros_and_actions_for_Q0.0001_steps_1000.npz", help="Name of the observation file)")
    parser.add_argument("--initial_Q", type=float, default="0.0001", help="initial Q to start the calibration")
    parser.add_argument("--person_number", type=str, default="person_0.0001", help="Person identifier for saving results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting the analysis..")
    # === Run your pipeline ===
    main(file_path=args.file_path, initial_Q=args.initial_Q , person_number=args.person_number)
