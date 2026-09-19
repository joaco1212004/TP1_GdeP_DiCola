import numpy as np
import torch
import os
import gymnasium as gym
from reinforce import observation_to_tensor


def evaluate_random_agent(env, episodes=100):
    returns = []
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()

        terminated = False
        truncated = False

        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)

    return np.mean(returns), np.mean(lengths)


def evaluate_fixed_action(env, action, episodes=100):
    returns = []
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()

        terminated = False
        truncated = False

        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)

    return np.mean(returns), np.mean(lengths)


def evaluate_policy(env, policy, episodes=100):
    returns = []
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()

        terminated = False
        truncated = False

        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            state = observation_to_tensor(
                obs,
                env.observation_space
            )

            with torch.no_grad():
                logits = policy(state)
                action = torch.argmax(logits).item()

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)

    return (
        np.mean(returns),
        np.mean(lengths),
        returns,
        lengths,
    )

def record_episode(env_name, policy, video_folder, name_prefix):
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(
        env_name,
        render_mode="rgb_array"
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: True,
    )

    obs, _ = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        state = observation_to_tensor(
            obs,
            env.observation_space
        )

        with torch.no_grad():
            logits = policy(state)
            action = torch.argmax(logits).item()

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1

    print("Total reward:", total_reward)
    print("Episode length:", steps)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()