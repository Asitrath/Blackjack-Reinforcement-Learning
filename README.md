# Blackjack Reinforcement Learning

This project implements various reinforcement learning algorithms to solve the Blackjack-v1 environment from OpenAI Gym.

![blackjack1](https://github.com/user-attachments/assets/9852fd7d-d26d-4248-b8e2-569921ffc0c6)

## Description

This project explores different reinforcement learning techniques applied to the game of Blackjack. It includes implementations of:

1. A simple policy that sticks for player sum >= 20
2. First-visit Monte Carlo prediction for policy evaluation
3. Monte Carlo Exploring Starts (ES) for finding an optimal policy

## Features

- Single run simulation of a basic Blackjack strategy
- Policy evaluation using first-visit Monte Carlo prediction
- Policy improvement using Monte Carlo ES
- Visualization of the value function for states with and without a usable ace

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenAI Gym

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/blackjack-rl.git
cd blackjack-rl
pip install numpy matplotlib gym
```

## Usage

Run the main script to execute all implemented algorithms:

```bash
python blackjack_rl.py
```

## Implementation Details

- `single_run_20()`: Simulates a single episode using a simple policy
- `policy_evaluation()`: Implements first-visit Monte Carlo prediction
- `monte_carlo_es()`: Implements Monte Carlo Exploring Starts for policy improvement
- `generate_episode_from_policy()`: Generates an episode following a given policy
- `plot_value_function()`: Visualizes the learned value function

## Results

The script outputs:
- The optimal policy found by Monte Carlo ES for states with and without a usable ace
- 3D plots of the value function for states with and without a usable ace
  
![Screenshot 2024-05-17 215303](https://github.com/user-attachments/assets/e33f647c-a5be-427f-9926-01d86af024aa)

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the Blackjack-v1 environment from OpenAI Gym and implements reinforcement learning algorithms as described in the literature.
```

This README provides an overview of your Blackjack reinforcement learning project, including what it does, how to set it up, how to use it, and what results to expect. You may want to customize it further based on any specific details or additional information you'd like to include.
