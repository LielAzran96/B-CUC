
import numpy as np
import torch
import pandas as pd
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params = {
#     'A' : np.array([[-0.5, -0.5], [-0.5, -1]], dtype=np.float64), # State transition matrix - position and velocity model   eigenvalues -1.3, -0.19
#     'B' : np.array([0,1],dtype=np.float64).reshape(2,1), # Control matrix (affects only velocity)
#     'Q' : np.diag([0.5, 0.5]).astype(np.float64),  # Process noise covariance
#     'R' : np.eye(2) * 0,   # Measurement noise covariance
#     'H' : np.eye(2),     # Observation matrix
#     'Q_cost' : np.array([[50, 0], [0, 1]]), # State cost matrix
#     'R_cost' : np.array([1]),
#     'x_goal' : np.array([10, 0]).reshape(2, 1),
#     'time_step': 0.1,
#     'max_episode_time_steps' : 100,
#     'initial_mean' : np.array([1,1]),
#     'initial_cov' : np.diag([0.5,0.5]),
#     'device' : device
# }

params = {
    'A': np.array([[np.log(0.95)]], dtype=np.float64),  # Slight decay over time (drug clearance)
    'B': np.array([[0.1]], dtype=np.float64),   # Dosage affects concentration directly
    # 'Q': np.array([[1.5]], dtype=np.float64),  
    'Q': np.array([[0.6275]], dtype=np.float64),  

    'R': np.array([[0.0001]], dtype=np.float64),  # Measurement noise in blood test
    'H': np.array([[1.0]]),                     # Direct observation of concentration
    'Q_cost': np.array([[5.0]]),                # Penalize deviation from target
    'R_cost': np.array([0.1]),                  # Penalize large doses
    'x_goal': np.array([[1.0]]),                # Target concentration (e.g., therapeutic level)
    'time_step': 0.1,
    'max_episode_time_steps': 100,
    'initial_mean': np.array([10]),            # Start with no drug in system
    'initial_cov': np.array([[0.1]]),           # Some uncertainty
    'device': device
}


def generate_observation_sequence(model, actions_sequence):
    """
    Generate a sequence of observations given a sequence of actions for any model
    
    Args:
        model: The model instance (should have reset(), step(), and time attributes)
        actions_sequence: List or array of actions to apply
        save_intermediate_states: Whether to store intermediate states
        
    Returns:
        dict: Contains observations, states (if requested), and timestamps
    """
    observations = []
    timestamps = []
    
    # Reset environment
    initial_obs = model.reset() 
    
    observations.append(initial_obs)
    timestamps.append(model.time)
    
    
    # Step through actions
    for action in actions_sequence:
        obs = model.step(action)
        observations.append(obs)
        timestamps.append(model.time)

    actions_sequence = np.concatenate((np.array([0]).reshape(-1,1), actions_sequence))  # Prepend initial action (0 for no action at start)
    result = {
        'observations': np.array(observations),
        'timestamps': np.array(timestamps),
        'actions': np.array(actions_sequence)
    }
        
    return result


def save_observations(data, filename, format='npz', dir_name=None):
    """
        Save observation data in various formats
        
        Args:
            data: Dictionary containing observations and metadata
            filename: Name of file to save (without extension)
            format: Format to save in ('npz')
            dir_name: Directory to save the file in (optional)
        """    
        
    if format == 'npz':
        # NumPy compressed format - best for numerical data
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            filepath = f"{dir_name}/{filename}.npz"
            params_filepath = f"{dir_name}/{filename}_params.json"
        else:
            filepath = f"{filename}.npz"
            params_filepath = f"{filename}_params.json"
        
        np.savez_compressed(filepath, **data)
        print(f"Data saved as {filepath}")
        
        # Save model parameters in JSON format
        model_params = {
            'A': params['A'].tolist(),
            'B': params['B'].tolist(),
            'Q': params['Q'].tolist(),
            'R': params['R'].tolist(),
            'H': params['H'].tolist()
        }
        
        with open(params_filepath, 'w') as f:
            json.dump(model_params, f, indent=2)
        print(f"Model parameters saved as {params_filepath}")
            
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_observations(filename, format='npz', dir_name=None):
    """
    Load observation data from file
    
    Args:
        filename: Name of file to load (without extension)
        format: Format to load from ('npz')
        
    Returns:
        dict: Dictionary containing observations and metadata
    """
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        filepath = f"{dir_name}/{filename}.npz"
        params_filepath = f"{dir_name}/{filename}_params.json"
    else:
        filepath = f"{filename}.npz"
        params_filepath = f"{filename}_params.json"
    if format == 'npz':
        # Load NumPy compressed format
        loaded = np.load(filepath)
        data = {key: loaded[key] for key in loaded.files}
        print(f"Data loaded from {filepath}")
        return data
        
    else:
        raise ValueError(f"Unsupported format: {format}")



# def load_and_analyze_example():
#     """Example of how to load and analyze saved data"""
    
#     print("\n" + "="*50)
#     print("Loading and analyzing saved data...")
    
#     # Load NPZ file
#     filename = "observations_sinusoidal.npz"
#     try:
#         loaded_data = np.load(filename)
        
#         print(f"\nLoaded data from {filename}")
#         print("Available keys:", list(loaded_data.keys()))
        
#         observations = loaded_data['observations']
#         timestamps = loaded_data['timestamps']
#         actions = loaded_data['actions']
#         states = loaded_data['states'] if 'states' in loaded_data else None
        
#         print(f"Observations shape: {observations.shape}")
#         print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f}")
#         print(f"Actions range: {actions.min():.3f} to {actions.max():.3f}")
        
#         if states is not None:
#             print(f"States shape: {states.shape}")

    # except FileNotFoundError:
    #         print(f"File {filename} not found. Run the main generation first.")

def mask_observations(observations, mask_probability=0.3):
    """
    Randomly mask a percentage of observations
    
    Args:
        observations: Array of observations to mask
        mask_probability: Probability of each observation being masked (default 30%)
        
    Returns:
        np.ndarray: Masked observations with some set to None
    """
    np.random.seed(42)  # For reproducibility
    random_mask = np.random.random(len(observations)) < mask_probability
    masked_observations = observations.copy()
    masked_observations[random_mask] = None  # Set masked observations to None
    return masked_observations