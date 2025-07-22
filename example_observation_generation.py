#!/usr/bin/env python3
"""
Example script showing how to generate and save observation sequences
using the LinearModel class.
"""

import numpy as np
import torch
from models.data_generator.data_generator import LinearModel

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
            num_zeros = np.random.randint(1, 10)
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

# def visualize_time_alignment():
#     """Demonstrate the importance of proper time alignment"""
    
#     print("\n" + "="*60)
#     print("TIME ALIGNMENT DEMONSTRATION")
#     print("="*60)
    
#     time_step = 0.1
#     n_steps = 50
    
#     # Correct time vector (what your model uses)
#     correct_time = np.arange(n_steps) * time_step
    
#     # Wrong time vector (what was used before)
#     wrong_time = np.linspace(0, 4*np.pi, n_steps)
    
#     print(f"Model time step: {time_step}")
#     print(f"Number of steps: {n_steps}")
#     print(f"Total simulation time: {n_steps * time_step} seconds")
#     print()
    
#     print("CORRECT approach:")
#     print(f"  Time vector: [0, {time_step}, {2*time_step}, ..., {(n_steps-1)*time_step}]")
#     print(f"  First 5 values: {correct_time[:5]}")
#     print(f"  Last 5 values: {correct_time[-5:]}")
#     print()
    
#     print("WRONG approach (old code):")
#     print(f"  Time vector: linspace(0, 4π, {n_steps}) = [0, ..., {4*np.pi:.2f}]")
#     print(f"  First 5 values: {wrong_time[:5]}")
#     print(f"  Last 5 values: {wrong_time[-5:]}")
#     print()
    
#     # Generate sinusoidal actions with both approaches
#     frequency = 2 * np.pi / (n_steps * time_step / 2)  # 2 cycles over total time
    
#     correct_actions = 0.5 * np.sin(frequency * correct_time)
#     wrong_actions = 0.5 * np.sin(wrong_time)
    
#     print("ACTION COMPARISON:")
#     print("Correct actions (first 10):", correct_actions[:10])
#     print("Wrong actions (first 10):  ", wrong_actions[:10])
#     print()
#     print("The correct approach ensures that:")
#     print("1. Actions are synchronized with model's internal time")
#     print("2. The frequency is appropriate for the simulation duration")
#     print("3. The control signal makes physical sense")

def main():
    """Main function to demonstrate observation generation and saving"""
    print("Creating linear system...")
    system = create_example_system()
    
    # Generate different types of action sequences
    n_steps = 200
    # action_types = ['random_zeros_and_actions', 'random_actions']
    action_types = ['random_zeros_and_actions']
    
    for action_type in action_types:
        print(f"\nGenerating observations for {action_type} actions...")
        
        # Generate actions with proper time step
        actions = generate_sample_actions(n_steps, action_type, time_step=system.time_step)
        
        # Generate observation sequence
        data = generate_observation_sequence(system, actions)
        
        print(f"Generated {len(data['observations'])} observations")
        print(f"Observation shape: {data['observations'][0].shape}")
        
        # Save in different formats
        person_name = "first_person"  # Change this for different people
        save_dir = f"observations/{person_name}"
        base_filename = f"observations_{action_type}_for_Q{system.Q.item()}_initialModel_1.5"
        
        # Save as compressed NumPy format (recommended for numerical data)
        save_observations(data, base_filename, format='npz', dir_name=save_dir)
        
        # # Save as CSV (good for analysis in Excel/pandas)
        # save_observations(data, base_filename, format='csv')
        
        # Optional: Save as JSON (human readable but larger files)
        # system.save_observations(data, base_filename, format='json')
        
        # Print some statistics
        obs_mean = np.mean(data['observations'], axis=0)
        obs_std = np.std(data['observations'], axis=0)
        print(f"Observation mean: {obs_mean.flatten()}")
        print(f"Observation std: {obs_std.flatten()}")


            
        # # Basic analysis
        # print(f"\nObservation statistics:")
        # print(f"  Mean: {np.mean(observations, axis=0).flatten()}")
        # print(f"  Std:  {np.std(observations, axis=0).flatten()}")
        
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # # Demonstrate time alignment importance
    # visualize_time_alignment()
    
    # Generate observations
    main()
    
    # Load and analyze
    # load_and_analyze_example()
    person_name = "first_person"  # Change this for different people
    save_dir = f"observations/{person_name}"
    base_filename = f"observations_random_zeros_and_actions_for_Q0_initialModel_0"
    load_observations(base_filename, format='npz', dir_name=save_dir)    
    print("\n" + "="*50)
    print("Observation generation complete!")
    print("\nRecommended workflow:")
    print("1. Use .npz format for numerical analysis in Python")
    print("2. Use .csv format for analysis in Excel or pandas")
    print("3. Use .json format for human-readable debugging")
    print("4. Use .mat format for MATLAB interoperability")
