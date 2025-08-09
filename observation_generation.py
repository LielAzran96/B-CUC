#!/usr/bin/env python3
"""
Example script showing how to generate and save observation sequences
using the LinearModel class.
"""

import numpy as np
import torch
from models.data_generator.data_generator import LinearModel
import argparse

from models.model_utils import params, generate_observation_sequence, save_observations, load_observations, mask_observations

def create_example_system():
    """Create an example 2D linear system"""

    return LinearModel(**params)

def generate_sample_actions(n_steps, action_type='random_zeros_and_actions', time_step=0.1):
    """Generate different types of action sequences"""
    
    # Create proper time vector based on the model's time step
    t = np.arange(n_steps) * time_step
    
    if action_type == 'random_zeros_and_actions':
        # Start with first action = 20
        actions = np.zeros(n_steps)
        actions[0] = 20
        
        i = 1
        while i < n_steps:
            # Random number of zeros (1-5)
            num_zeros = np.random.randint(1, 20)
            for j in range(num_zeros):
                if i < n_steps:
                    actions[i] = 0
                    i += 1
            
            # Random positive action in range (0, 30]
            if i < n_steps:
                actions[i] = np.random.uniform(0.1, 30)
                i += 1
           
        
    elif action_type == 'step':
        # Step function - modified to be non-negative
        actions = np.ones(n_steps) * 0.3
        actions[n_steps//2:] = 0.1
        
    elif action_type == 'random_non_negative':
        # Random actions - modified to be non-negative
        np.random.seed(42)
        actions = np.abs(np.random.normal(0.5, 0.3, n_steps))
        
    elif action_type == 'ramp':
        # Ramp function - modified to be non-negative
        actions = np.linspace(0, 1.0, n_steps)
        
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    
    return actions.reshape(-1, 1)

'''
for debug usage
'''
# def main():
#     """Main function to demonstrate observation generation and saving"""
#     print("Creating linear system...")
#     system = create_example_system()
    
#     # Generate different types of action sequences
#     n_steps = 200
#     # action_types = ['random_zeros_and_actions', 'random_actions']
#     action_types = ['random_zeros_and_actions']
    
#     for action_type in action_types:
#         print(f"\nGenerating observations for {action_type} actions...")
        
#         # Generate actions with proper time step
#         actions = generate_sample_actions(n_steps, action_type, time_step=system.time_step)
        
#         # Generate observation sequence
#         data = generate_observation_sequence(system, actions)
        
#         print(f"Generated {len(data['observations'])} observations")
#         print(f"Observation shape: {data['observations'][0].shape}")
        
#         # Save in different formats
#         person_name = "first_person"  # Change this for different people
#         save_dir = f"observations/{person_name}"
#         base_filename = f"observations_{action_type}_for_Q{system.Q.item()}_initialModel_1.5"
        
#         # Save as compressed NumPy format (recommended for numerical data)
#         save_observations(data, base_filename, format='npz', dir_name=save_dir)
        
#         # # Save as CSV (good for analysis in Excel/pandas)
#         # save_observations(data, base_filename, format='csv')
        
#         # Optional: Save as JSON (human readable but larger files)
#         # system.save_observations(data, base_filename, format='json')
        
#         # Print some statistics
#         obs_mean = np.mean(data['observations'], axis=0)
#         obs_std = np.std(data['observations'], axis=0)
#         print(f"Observation mean: {obs_mean.flatten()}")
#         print(f"Observation std: {obs_std.flatten()}")



def main(dir_path, initial_Q , action_type, file_format):
    """Main function to demonstrate observation generation and saving"""
    print("Creating linear system...")
    system = create_example_system()
    print(system.Q)
    
    # Generate different types of action sequences
    n_steps = 1000
   
    print(f"\nGenerating observations for {action_type} actions...")
    
    # Generate actions
    actions = generate_sample_actions(n_steps, action_type, time_step=system.time_step)
    
    # Generate observation sequence
    data = generate_observation_sequence(system, actions)
    
    print(f"Generated {len(data['observations'])} observations")
    print(f"Observation shape: {data['observations'][0].shape}")
    
    # Final save_dir fallback
    save_dir = f"{dir_path}"
    base_filename = f"observations_{action_type}_for_Q{system.Q.item()}_initialModel_{initial_Q}"
    
    # Save observations
    save_observations(data, base_filename, format=file_format, dir_name=save_dir)
    
    # Print stats
    obs_mean = np.mean(data['observations'], axis=0)
    obs_std = np.std(data['observations'], axis=0)
    print(f"Observation mean: {obs_mean.flatten()}")
    print(f"Observation std: {obs_std.flatten()}")
       
def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive calibration pipeline.")
    parser.add_argument("--dir_path", type=str, default="observations/first_person", help="Name of the person (used for folder naming)")
    parser.add_argument("--action_type", type=str, default="random_zeros_and_actions", help="Action type for observation generation")
    parser.add_argument("--initial_Q", type=str, default="0.0001", help="Initial Q value")
    parser.add_argument("--format", type=str, default="npz", choices=["npz", "csv", "json", "mat"], help="File format to load")
    return parser.parse_args()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()

    # === Run your pipeline ===
    # Step 1: Generate observations (this should save files in save_dir)
    main(dir_path=args.dir_path, initial_Q=args.initial_Q , action_type=args.action_type, file_format=args.format)

    print("Observation generation complete!")
    print("\nRecommended workflow:")
    print("1. Use .npz format for numerical analysis in Python")
    print("2. Use .csv format for analysis in Excel or pandas")
    print("3. Use .json format for human-readable debugging")
    print("4. Use .mat format for MATLAB interoperability")


'''
    for debug usage only
'''

# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     # # Demonstrate time alignment importance
#     # visualize_time_alignment()
    
#     # Generate observations
#     main()
    
#     # Load and analyze
#     # load_and_analyze_example()
#     person_name = "first_person"  # Change this for different people
#     save_dir = f"observations/{person_name}"
#     base_filename = f"observations_random_zeros_and_actions_for_Q0_initialModel_0"
#     load_observations(base_filename, format='npz', dir_name=save_dir)    
#     print("\n" + "="*50)
#     print("Observation generation complete!")
#     print("\nRecommended workflow:")
#     print("1. Use .npz format for numerical analysis in Python")
#     print("2. Use .csv format for analysis in Excel or pandas")
#     print("3. Use .json format for human-readable debugging")
#     print("4. Use .mat format for MATLAB interoperability")
