from models.linear_model import LinearDynamics
from models.linear_model import LinearEnvWrapper
import numpy as np
import traceback
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    'A' : np.array([[-0.5, -0.5], [-0.5, -1]], dtype=np.float64), # State transition matrix - position and velocity model   eigenvalues -1.3, -0.19
    'B' : np.array([0,1],dtype=np.float64).reshape(2,1), # Control matrix (affects only velocity)
    'Q' : np.diag([0.5, 0.5]).astype(np.float64),  # Process noise covariance
    'R' : np.eye(2) * 0,   # Measurement noise covariance
    'H' : np.eye(2),     # Observation matrix
    'Q_cost' : np.array([[50, 0], [0, 1]]), # State cost matrix
    'R_cost' : np.array([1]),
    'x_goal' : np.array([10, 0]).reshape(2, 1),
    'time_step': 0.1,
    'max_episode_time_steps' : 100,
    'initial_mean' : np.array([1,1]),
    'initial_cov' : np.diag([0.5,0.5]),
    'device' : device
}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# params = {
#     'device': device,
#     'A': torch.tensor([[-0.5, -0.5], [-0.5, -1.0]], dtype=torch.float32, device = device),
#     'B': torch.tensor([[0.0], [1.0]], dtype=torch.float32, device = device),  # Control matrix (affects only velocity)
#     'Q': torch.diag(torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)),  # Process noise covariance
#     'R': torch.eye(2, dtype=torch.float32, device=device) * 0.00001,
#     'H': torch.eye(2, dtype=torch.float32, device=device),  # Observation matrix
#     'Q_cost': torch.tensor([[50.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device),  # State cost matrix
#     'R_cost': torch.tensor([1.0], dtype=torch.float32, device=device),  # scalar can stay as-is
#     'x_goal': torch.tensor([10.0, 0.0], dtype=torch.float32, device=device).reshape(2, 1),
#     'time_step': 0.1,  # scalar
#     'max_episode_time_steps': 40,  # integer
#     'initial_mean': torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
#     'initial_cov': torch.diag(torch.tensor([0.5, 0.5], dtype=torch.float32, device=device))  # Initial covariance matrix,
# }

params_for_wrapper = {
    'sample_cost' : 0.1,
    'const_noise' : True
}

