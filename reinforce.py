import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models import PolicyNetwork, ValueNetwork


def observation_to_tensor(obs, observation_space):
    """
    Convierte una observación del entorno en tensor para la red.

    - Si el espacio es Discrete, usa one-hot encoding.
    - Si es continuo, convierte directamente a float tensor.
    """
    if hasattr(observation_space, "n"):
        x = F.one_hot(
            torch.tensor(int(obs)),
            num_classes=observation_space.n
        ).float()
    else:
        x = torch.tensor(obs, dtype=torch.float32)

    return x


def discounted_returns(rewards, gamma=0.99):
    """
    Calcula el reward-to-go:
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    returns.reverse()

    return returns


def collect_episode(env, policy):
    """
    Ejecuta un episodio completo usando la policy estocástica.
    """
    obs, _ = env.reset()

    log_probs = []
    rewards = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = observation_to_tensor(
            obs,
            env.observation_space
        )

        logits = policy(state)

        dist = Categorical(logits=logits)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(
            action.item()
        )

        log_probs.append(log_prob)
        rewards.append(float(reward))

        obs = next_obs

    return log_probs, rewards


def train_reinforce(
    env,
    num_episodes=500,
    batch_size=10,
    gamma=0.99,
    learning_rate=1e-3,
    normalize_returns=True,
):
    """
    Entrena una política usando REINFORCE.
    """

    if hasattr(env.observation_space, "n"):
        input_dim = env.observation_space.n
    else:
        input_dim = env.observation_space.shape[0]

    output_dim = env.action_space.n

    policy = PolicyNetwork(
        input_dim=input_dim,
        output_dim=output_dim
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate
    )

    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    losses = []
    loss_steps = []

    episodes_done = 0

    while episodes_done < num_episodes:

        batch_log_probs = []
        batch_returns = []

        batch_episode_rewards = []
        batch_episode_lengths = []

        episodes_in_batch = min(
            batch_size,
            num_episodes - episodes_done
        )

        for _ in range(episodes_in_batch):
            log_probs, rewards = collect_episode(
                env,
                policy
            )

            returns = discounted_returns(
                rewards,
                gamma=gamma
            )

            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

            total_reward = sum(rewards)

            batch_episode_rewards.append(total_reward)
            batch_episode_lengths.append(len(rewards))

        returns_tensor = torch.tensor(
            batch_returns,
            dtype=torch.float32
        )

        if normalize_returns and len(returns_tensor) > 1:
            returns_tensor = (
                returns_tensor - returns_tensor.mean()
            ) / (
                returns_tensor.std() + 1e-8
            )

        log_probs_tensor = torch.stack(batch_log_probs)

        weighted_terms = -(
            log_probs_tensor * returns_tensor
        )

        episode_loss_tensors = []
        offset = 0

        for episode_length in batch_episode_lengths:
            episode_loss = weighted_terms[
                offset:offset + episode_length
            ].sum()

            episode_loss_tensors.append(episode_loss)

            offset += episode_length

        loss = torch.stack(
            episode_loss_tensors
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.extend(batch_episode_rewards)
        episode_lengths.extend(batch_episode_lengths)

        episode_losses.extend(
            [
                episode_loss.detach().item()
                for episode_loss in episode_loss_tensors
            ]
        )

        episodes_done += episodes_in_batch

        losses.append(loss.item())
        loss_steps.append(episodes_done)

    history = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "episode_losses": episode_losses,
        "losses": losses,
        "loss_steps": loss_steps,
    }

    return policy, history

def collect_episode_with_states(env, policy):
    obs, _ = env.reset()

    states = []
    log_probs = []
    rewards = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = observation_to_tensor(
            obs,
            env.observation_space
        )

        logits = policy(state)

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(
            action.item()
        )

        states.append(state)
        log_probs.append(log_prob)
        rewards.append(float(reward))

        obs = next_obs

    return states, log_probs, rewards

def train_reinforce_with_baseline(
    env,
    num_episodes=500,
    batch_size=10,
    gamma=0.99,
    policy_learning_rate=1e-3,
    value_learning_rate=1e-3,
    value_epochs=5,
):
    if hasattr(env.observation_space, "n"):
        input_dim = env.observation_space.n
    else:
        input_dim = env.observation_space.shape[0]

    output_dim = env.action_space.n

    policy = PolicyNetwork(
        input_dim=input_dim,
        output_dim=output_dim
    )

    value_network = ValueNetwork(
        input_dim=input_dim
    )

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=policy_learning_rate
    )

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=value_learning_rate
    )

    episode_rewards = []
    episode_lengths = []
    episode_policy_losses = []
    value_losses = []

    episodes_done = 0

    while episodes_done < num_episodes:

        batch_states = []
        batch_log_probs = []
        batch_returns = []
        batch_episode_lengths = []

        batch_episode_rewards = []

        episodes_in_batch = min(
            batch_size,
            num_episodes - episodes_done
        )

        for _ in range(episodes_in_batch):
            states, log_probs, rewards = collect_episode_with_states(
                env,
                policy
            )

            returns = discounted_returns(
                rewards,
                gamma=gamma
            )

            batch_states.extend(states)
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

            batch_episode_lengths.append(len(rewards))
            batch_episode_rewards.append(sum(rewards))

        states_tensor = torch.stack(batch_states)

        returns_tensor = torch.tensor(
            batch_returns,
            dtype=torch.float32
        )

        log_probs_tensor = torch.stack(
            batch_log_probs
        )

        # V_phi(s_t)
        values = value_network(states_tensor)

        # A(t) = R(t) - V_phi(s_t)
        advantages = (
            returns_tensor - values.detach()
        )

        if len(advantages) > 1:
            advantages_for_policy = (
                advantages - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )
        else:
            advantages_for_policy = advantages

        # Policy loss:
        # -(1/N) sum_i sum_t log pi(a_t|s_t) A_t
        weighted_terms = -(
            log_probs_tensor * advantages_for_policy
        )

        episode_loss_tensors = []
        offset = 0

        for episode_length in batch_episode_lengths:
            episode_loss = weighted_terms[
                offset:offset + episode_length
            ].sum()

            episode_loss_tensors.append(
                episode_loss
            )

            offset += episode_length

        policy_loss = torch.stack(
            episode_loss_tensors
        ).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Value network:
        # minimizar (V_phi(s_t) - R(t))^2
        for _ in range(value_epochs):
            values_for_fit = value_network(states_tensor)

            value_loss = torch.mean(
                (values_for_fit - returns_tensor) ** 2
            )

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        value_losses.append(
            value_loss.item()
        )

        episode_rewards.extend(
            batch_episode_rewards
        )

        episode_lengths.extend(
            batch_episode_lengths
        )

        episode_policy_losses.extend(
            [
                loss.detach().item()
                for loss in episode_loss_tensors
            ]
        )

        episodes_done += episodes_in_batch

    history = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "episode_losses": episode_policy_losses,
        "value_losses": value_losses,
    }

    return policy, value_network, history