import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")
parser.add_argument("--env", default="HalfCheetah-v4")
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--size", default=0, type=float)
args = parser.parse_args()
env = gym.make(args.env)
env.seed(2)
env.action_space.seed(2)
torch.manual_seed(2)
np.random.seed(2)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": 0.99,
    "tau": 0.005,
}

kwargs["policy_noise"] = 0.2 * max_action
kwargs["noise_clip"] = 0.5 * max_action
kwargs["policy_freq"] = 2
policy = TD3.TD3(**kwargs)

policy.load(f"./models/TD3_{args.env}_{args.seed}")
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)


def rollout(policy: TD3.TD3, replay_buffer, env):
    state, done = env.reset(), False
    episode_reward = 0
    device = policy.actor.parameters().__next__().device
    for t in range(1000):
        with torch.no_grad():
            action = policy.select_action(np.array(state))
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done)
        with torch.no_grad():
            Q_value = reward + 0.99 * \
                policy.critic.Q1(torch.tensor(next_state, dtype=torch.float32).to(
                    device).reshape(1, -1), torch.tensor(action, dtype=torch.float32).to(device).reshape(1, -1)).cpu().numpy().flatten()
        replay_buffer.add(state, action, next_state,
                          reward, done_bool, Q_value)
        state = next_state
        episode_reward += reward
        if done:
            break


while replay_buffer.size < args.size:
    rollout(policy, replay_buffer, env)


replay_buffer.save(f"TD3_{args.env}_data")
