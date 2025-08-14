# BCUC – Final Project in Reliability in Machine Learning

This repository contains the final project for the course **"Reliability in Machine Learning"** at the **Technion – Israel Institute of Technology**.

The project is inspired by the paper **CUQDS: Conformal Uncertainty Quantification under Distribution Shift**, which proposes a framework for uncertainty calibration of trajectories, under distribution shift. Our work adapts and extends some of these ideas to a setting where the process noise `Q` in a dynamical system can be adaptively calibrated over time.

> 📄 **Referencing CUQDS**  
> ```
> @inproceedings{huang2025cuqds,
>  title={CUQDS: Conformal Uncertainty Quantification Under Distribution Shift for Trajectory Prediction},
>  author={Huang, Huiqun and He, Sihong and Miao, Fei},
>  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>  volume={39},
>  number={16},
>  pages={17422--17430},
>  year={2025}
>}
> ```
---

## 📦 Environment Setup

We recommend using a Python virtual environment to keep dependencies isolated.

```bash
# Create a virtual environment (Python 3.x)
python3 -m venv .venv

# Activate the environment (Mac/Linux)
source .venv/bin/activate

# Activate the environment (Windows PowerShell)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 📊 Observation Generation

The observations/ directory contains several pre-generated observation datasets.
Each file corresponds to:

1. A different process noise value (Q)
2. A different number of timesteps

If you want to generate new observations, use the script:

```bash
./run_observation_generation.sh [--initial_Q INITIAL_Q] [--n_steps N_STEPS]
```

## 📈 Performance Analysis

The performance analysis script evaluates the behavior of the BCUC algorithm using generated observations.
First, change directory to final_algorithm.
Then, run it with:
```bash
./run_performance.sh [--initial_Q INITIAL_Q]
```
Pay attention it uses 4 different Q for comparison. If you wish to see the comparison with different Q you generated, just change it inside the performance.py.

## 🔍 Comparison Analysis

The comparison script compares BCUC against other methods.
First, change directory to final_algorithm/Comparison
Then, run it with:
```bash
./run_comparison.sh [--initial_Q INITIAL_Q]
```

## 💥 q_Q Explosion Analysis

The q_Q explosion analysis script evaluates how q_t and Q_t evolve over time and whether they diverge, with the first try of the implementation of the algorithm.
First, change directory to q_Q_explosion_analysis
Then, run it with:
```bash
./run_q_Q_explosion_analysis.sh [--dir_path FILE_PATH] [--initial_Q INITIAL_Q] [--person_number PERSON_NUMBER]
```

