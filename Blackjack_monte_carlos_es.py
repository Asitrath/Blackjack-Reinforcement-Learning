import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial

plt.style.use('ggplot')

env = gym.make('Blackjack-v1')

def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks

    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        #print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            #print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            #print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        #print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    #print("final observation:", obs)
    return states, ret


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 10000 # use whatever number of iterations you want
    for i in range(maxiter):
        episode_states, episode_return = single_run_20()
        visited_states = set()
        for state in episode_states:
            player_sum, dealer_card, useable_ace = state
            if (player_sum, dealer_card, useable_ace) not in visited_states:
                returns[player_sum-12, dealer_card-1, int(useable_ace)] += episode_return
                visits[player_sum-12, dealer_card-1, int(useable_ace)] += 1
                V[player_sum-12, dealer_card-1, int(useable_ace)] = returns[player_sum-12, dealer_card-1, int(useable_ace)] / visits[player_sum-12, dealer_card-1, int(useable_ace)]
                visited_states.add((player_sum, dealer_card, useable_ace))
    return V


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2), dtype=int)
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 5000000  # use whatever number of iterations you want

    for i in range(maxiter):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print("Policy (no usable ace):")
            print(pi[:, :, 0])
            print("Policy (usable ace):")
            print(pi[:, :, 1])

        # Generate an episode using the current policy
        states, actions, rewards = generate_episode_from_policy(pi)

        G = 0
        visited = set()
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            G += rewards[t]
            state_action = (state[0] - 12, state[1] - 1, int(state[2]), action)

            if state_action not in visited:
                returns[state_action] += G
                visits[state_action] += 1
                Q[state_action] = returns[state_action] / visits[state_action]
                pi[state_action[:-1]] = np.argmax(Q[state[0] - 12, state[1] - 1, int(state[2]), :])
                visited.add(state_action)

    print("Optimal policy (no usable ace):")
    print(pi[:, :, 0])
    print("Optimal policy (usable ace):")
    print(pi[:, :, 1])


def generate_episode_from_policy(pi):
    """ run a single episode following policy pi """
    obs = env.reset()
    states = []
    actions = []
    rewards = []
    done = False
    while not done:
        states.append(obs)
        action = pi[obs[0] - 12, obs[1] - 1, int(obs[2])]
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    return states, actions, rewards

def plot_value_function(V,ax1,ax2):
    player_sum_range = np.arange(12, 22)
    dealer_card_range = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_card_range, player_sum_range)

    ax1.plot_wireframe(X, Y, V[:, :, 0])
    ax2.plot_wireframe(X, Y, V[:, :, 1])

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')

def main():
    single_run_20()
    #policy_evaluation()
    monte_carlo_es()


    value= policy_evaluation()

    fig, axes = pyplot.subplots(nrows=1, ncols =2, figsize=(15, 15),
    subplot_kw={'projection': '3d'})
    axes[0].set_title('value function without usable ace')
    axes[1].set_title('value function with usable ace')
    plot_value_function(value, axes[0], axes[1])

if __name__ == "__main__":
    main()
