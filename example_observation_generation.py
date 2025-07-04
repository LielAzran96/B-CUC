#!/usr/bin/env python3
"""
Example script showing how to generate and save observation sequences
using the LinearModel class.
"""

import numpy as np
import torch
from models.data_generator.data_generator import LinearModel

from models.model_utils import params, generate_observation_sequence, save_observations

def create_example_system():
    """Create an example 2D linear system"""

    return LinearModel(**params)

def generate_sample_actions(n_steps, action_type='sinusoidal', time_step=0.1):
    """Generate different types of action sequences"""
    
    # Create proper time vector based on the model's time step
    t = np.arange(n_steps) * time_step
    
    if action_type == 'sinusoidal':
        # Sinusoidal control input - frequency adjusted for time step
        # With time_step=0.1 and n_steps=50, total time = 5.0 seconds
        # This creates about 2 full cycles over the simulation
        frequency = 2 * np.pi / (n_steps * time_step / 2)  # 2 cycles over total time
        actions = 0.5 * np.sin(frequency * t) + 0.2 * np.cos(2 * frequency * t)
        
    elif action_type == 'step':
        # Step function
        actions = np.ones(n_steps) * 0.3
        actions[n_steps//2:] = -0.2
        
    elif action_type == 'random':
        # Random actions
        np.random.seed(42)
        actions = np.random.normal(0, 0.3, n_steps)
        
    elif action_type == 'ramp':
        # Ramp function
        actions = np.linspace(-0.5, 0.5, n_steps)
        
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    
    return actions.reshape(-1, 1)

def visualize_time_alignment():
    """Demonstrate the importance of proper time alignment"""
    
    print("\n" + "="*60)
    print("TIME ALIGNMENT DEMONSTRATION")
    print("="*60)
    
    time_step = 0.1
    n_steps = 50
    
    # Correct time vector (what your model uses)
    correct_time = np.arange(n_steps) * time_step
    
    # Wrong time vector (what was used before)
    wrong_time = np.linspace(0, 4*np.pi, n_steps)
    
    print(f"Model time step: {time_step}")
    print(f"Number of steps: {n_steps}")
    print(f"Total simulation time: {n_steps * time_step} seconds")
    print()
    
    print("CORRECT approach:")
    print(f"  Time vector: [0, {time_step}, {2*time_step}, ..., {(n_steps-1)*time_step}]")
    print(f"  First 5 values: {correct_time[:5]}")
    print(f"  Last 5 values: {correct_time[-5:]}")
    print()
    
    print("WRONG approach (old code):")
    print(f"  Time vector: linspace(0, 4π, {n_steps}) = [0, ..., {4*np.pi:.2f}]")
    print(f"  First 5 values: {wrong_time[:5]}")
    print(f"  Last 5 values: {wrong_time[-5:]}")
    print()
    
    # Generate sinusoidal actions with both approaches
    frequency = 2 * np.pi / (n_steps * time_step / 2)  # 2 cycles over total time
    
    correct_actions = 0.5 * np.sin(frequency * correct_time)
    wrong_actions = 0.5 * np.sin(wrong_time)
    
    print("ACTION COMPARISON:")
    print("Correct actions (first 10):", correct_actions[:10])
    print("Wrong actions (first 10):  ", wrong_actions[:10])
    print()
    print("The correct approach ensures that:")
    print("1. Actions are synchronized with model's internal time")
    print("2. The frequency is appropriate for the simulation duration")
    print("3. The control signal makes physical sense")

def main():
    """Main function to demonstrate observation generation and saving"""
    print("Creating linear system...")
    system = create_example_system()
    
    # Generate different types of action sequences
    n_steps = 50
    action_types = ['sinusoidal', 'step', 'random', 'ramp']
    
    for action_type in action_types:
        print(f"\nGenerating observations for {action_type} actions...")
        
        # Generate actions with proper time step
        actions = generate_sample_actions(n_steps, action_type, time_step=system.time_step)
        
        # Generate observation sequence
        data = generate_observation_sequence(system, actions)
        
        print(f"Generated {len(data['observations'])} observations")
        print(f"Observation shape: {data['observations'][0].shape}")
        
        # Save in different formats
        base_filename = f"observations_{action_type}"
        
        # Save as compressed NumPy format (recommended for numerical data)
        save_observations(data, base_filename, format='npz')
        
        # Save as CSV (good for analysis in Excel/pandas)
        save_observations(data, base_filename, format='csv')
        
        # Optional: Save as JSON (human readable but larger files)
        # system.save_observations(data, base_filename, format='json')
        
        # Print some statistics
        obs_mean = np.mean(data['observations'], axis=0)
        obs_std = np.std(data['observations'], axis=0)
        print(f"Observation mean: {obs_mean.flatten()}")
        print(f"Observation std: {obs_std.flatten()}")

def load_and_analyze_example():
    """Example of how to load and analyze saved data"""
    
    print("\n" + "="*50)
    print("Loading and analyzing saved data...")
    
    # Load NPZ file
    filename = "observations_sinusoidal.npz"
    try:
        loaded_data = np.load(filename)
        
        print(f"\nLoaded data from {filename}")
        print("Available keys:", list(loaded_data.keys()))
        
        observations = loaded_data['observations']
        timestamps = loaded_data['timestamps']
        actions = loaded_data['actions']
        states = loaded_data['states'] if 'states' in loaded_data else None
        
        print(f"Observations shape: {observations.shape}")
        print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f}")
        print(f"Actions range: {actions.min():.3f} to {actions.max():.3f}")
        
        if states is not None:
            print(f"States shape: {states.shape}")
            
        # Basic analysis
        print(f"\nObservation statistics:")
        print(f"  Mean: {np.mean(observations, axis=0).flatten()}")
        print(f"  Std:  {np.std(observations, axis=0).flatten()}")
        
    except FileNotFoundError:
        print(f"File {filename} not found. Run the main generation first.")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Demonstrate time alignment importance
    visualize_time_alignment()
    
    # Generate observations
    main()
    
    # Load and analyze
    load_and_analyze_example()
    
    print("\n" + "="*50)
    print("Observation generation complete!")
    print("\nRecommended workflow:")
    print("1. Use .npz format for numerical analysis in Python")
    print("2. Use .csv format for analysis in Excel or pandas")
    print("3. Use .json format for human-readable debugging")
    print("4. Use .mat format for MATLAB interoperability")
