import numpy as np
import torch
import gym
import argparse
import torch.nn as nn
import utils
import warnings
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="policy")
parser.add_argument("--env", default="HalfCheetah-v4")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--hidden", default=32, type=int)
parser.add_argument("--layer", default=2, type=int)
parser.add_argument("--size", default=1e5, type=float)
parser.add_argument("--device", default=0, type=int)
args = parser.parse_args()
hidden_dim = args.hidden
num_layer = args.layer
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
task = args.type
size = int(args.size)
env = gym.make(args.env)
seed = args.seed
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
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

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer.load(f"TD3_{args.env}_data", size)


def MLP(in_dim, out_dim, hidden_dim=32, num_layer=2):
    if num_layer == 2:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
    else:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


Q_value, action, reward, state, next_state, not_done = replay_buffer.Q_value[:replay_buffer.size], replay_buffer.action[:replay_buffer.size], replay_buffer.reward[
    :replay_buffer.size], replay_buffer.state[:replay_buffer.size], replay_buffer.next_state[:replay_buffer.size], replay_buffer.not_done[:replay_buffer.size]

Q_value = torch.tensor(np.array(Q_value), dtype=torch.float32)
action = torch.tensor(np.array(action), dtype=torch.float32)
reward = torch.tensor(np.array(reward), dtype=torch.float32)
state = torch.tensor(np.array(state), dtype=torch.float32)
next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
err_list = []

if task == "policy":
    total_variance = torch.var(action, unbiased=False, dim=0)
    total_variance = total_variance.norm(p=1)
    model = MLP(state.size(dim=1), action.size(dim=1), hidden_dim, num_layer)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train = state
    y_train = action
    X_test = state
    y_test = action

elif task == "value":
    total_variance = torch.var(Q_value, unbiased=False)
    model = MLP(state.size(dim=1) + action.size(dim=1),
                1, hidden_dim, num_layer)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.cat((state, action), dim=1)
    y_train = Q_value
    X_test = torch.cat((state, action), dim=1)
    y_test = Q_value

elif task == "model":
    total_variance = torch.var(next_state, unbiased=False, dim=0)
    total_variance = total_variance.norm(p=1)
    model = MLP(state.size(dim=1) + action.size(dim=1),
                state.size(dim=1), hidden_dim, num_layer)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.cat((state, action), dim=1)
    y_train = next_state
    X_test = torch.cat((state, action), dim=1)
    y_test = next_state

else:
    total_variance = torch.var(reward, unbiased=False) + 1e-8
    model = MLP(state.size(dim=1) + action.size(dim=1),
                1, hidden_dim, num_layer)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.cat((state, action), dim=1)
    y_train = reward
    X_test = torch.cat((state, action), dim=1)
    y_test = reward

model.to(device)

print("-----------------------------------")
print(f"Task: {task}")
print(f"Environment: {args.env}")
print(f"Seed: {seed}")
print(f"Hidden Dimension: {hidden_dim}")
print(f"Number of Layers: {num_layer}")
print("-----------------------------------")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
num_epoch = int(9e6 / size)
total_variance = 1e-8 if total_variance <= 1e-8 else total_variance
if total_variance is not float:
    total_variance = total_variance.item()
for epoch in range(num_epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    progress_bar = enumerate(train_loader)

    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        average_loss = total_loss / num_batches
        relative_loss = total_loss / (total_variance * num_batches)
        print_period = 10
    err_list.append(relative_loss)
    if epoch % print_period == 0:
        print(
            f"Epoch {epoch}/{num_epoch}, Training Loss: {average_loss:.6f}, Relative Loss: {relative_loss:.6f}")

model.eval()
total_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
relative_loss = total_loss / (total_variance.item() * len(test_loader))
err_list += [relative_loss, average_loss]
print(f"Test Loss: {average_loss:.6f}, Relative Loss: {relative_loss:.6f}")
np.savetxt(
    f"./Error_stats/{args.env}-{args.type}-{args.seed}-{hidden_dim}-{num_layer}-{size}.txt", np.array(err_list))
