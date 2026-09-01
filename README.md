# Joint UAV Trajectory and Power Control using Deep Reinforcement Learning

This repository contains the implementation and evaluation code for my
UCL MSc dissertation on energy-efficient UAV-assisted wireless
communications using Proximal Policy Optimisation (PPO).

The proposed framework jointly optimises:
- 3D UAV trajectory
- total transmit power
- user-level power allocation

The evaluation includes mixed-mobility scenarios, power-allocation
ablation, interference-aware training, random-seed robustness, and
generalisation to unseen mobility and denser building environments.

## Main Files

- `train.py` — PPO training
- `evaluate_and_plot.py` — main evaluation and visualisation
- `evaluate_generalisation.py` — unseen mobility and building-layout evaluation
- `evaluate_training_seed.py` — multi-seed robustness evaluation
- `compare_group_a_c.py` — user-level power-allocation ablation
- `compare_interference_allocation_2x2.py` — interference/allocation comparison
- `monitor_training.py` — training metric monitoring
- `envs/env_1uav_5user_power.py` — UAV communication environment

## Requirements

Python 3.9+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
