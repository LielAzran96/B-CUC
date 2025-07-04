
import numpy as np
import torch
import pandas as pd

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
    'A': np.array([[0.95]], dtype=np.float64),  # Slight decay over time (drug clearance)
    'B': np.array([[0.1]], dtype=np.float64),   # Dosage affects concentration directly
    'Q': np.array([[0.01]], dtype=np.float64),  # Small process noise
    'R': np.array([[0.05]], dtype=np.float64),  # Measurement noise in blood test
    'H': np.array([[1.0]]),                     # Direct observation of concentration
    'Q_cost': np.array([[5.0]]),                # Penalize deviation from target
    'R_cost': np.array([0.1]),                  # Penalize large doses
    'x_goal': np.array([[1.0]]),                # Target concentration (e.g., therapeutic level)
    'time_step': 0.1,
    'max_episode_time_steps': 100,
    'initial_mean': np.array([0.0]),            # Start with no drug in system
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


def save_observations(data, filename, format='npz'):
    """
    Save observation data in various formats
    
    Args:
        data: Dictionary containing observations and metadata
        filename: Name of file to save (without extension)
        format: Format to save in ('npz', 'csv', 'json', 'mat')
    """
    import os
    
    if format == 'npz':
        # NumPy compressed format - best for numerical data
        np.savez_compressed(f"{filename}.npz", **data)
        print(f"Data saved as {filename}.npz")
        
    elif format == 'csv':
        # CSV format - good for simple tabular data
        
        # Flatten observations for CSV
        obs_flat = data['observations'].reshape(len(data['observations']), -1)
        df_data = {
            'timestamp': data['timestamps'],
            'action': data['actions'].flatten() if data['actions'].ndim > 1 else data['actions']
        }
        
        # Add observation columns
        for i in range(obs_flat.shape[1]):
            df_data[f'obs_{i}'] = obs_flat[:, i]
            
        df = pd.DataFrame(df_data)
        df.to_csv(f"{filename}.csv", index=False)
        print(f"Data saved as {filename}.csv")
        
    elif format == 'json':
        # JSON format - human readable but less efficient for large arrays
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
                
        with open(f"{filename}.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Data saved as {filename}.json")
        
    elif format == 'mat':
        # MATLAB format - good for MATLAB interoperability
        from scipy.io import savemat
        savemat(f"{filename}.mat", data)
        print(f"Data saved as {filename}.mat")
        
    else:
        raise ValueError(f"Unsupported format: {format}")


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