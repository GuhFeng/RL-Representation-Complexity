import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.Q_value = np.zeros((max_size, 1))
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, Q_value=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.Q_value[self.ptr] = Q_value
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, if_Q_value=False):
        ind = np.random.randint(0, self.size, size=batch_size)
        if if_Q_value:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.Q_value[ind]).to(self.device)
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )

    def save(self, file_name):
        lst = ["Q_value", "action", "reward",
               "state", "next_state", "not_done"]
        lst_data = [self.Q_value, self.action, self.reward,
                    self.state, self.next_state, self.not_done]
        for i in range(6):
            np.save(
                f"./rollouts/{file_name}_{lst[i]}.npy", lst_data[i][:self.size])

    def load(self, file_name, size=int(2e6)):
        lst = ["Q_value", "action", "reward",
               "state", "next_state", "not_done"]
        lst_data = [self.Q_value, self.action, self.reward,
                    self.state, self.next_state, self.not_done]
        indx = np.arange(size)
        np.random.shuffle(indx)
        for i in range(6):
            lst_data[i][:size] = np.load(
                f"./rollouts/{file_name}_{lst[i]}.npy")[indx[:size]]
        self.size = size
        self.ptr = size
