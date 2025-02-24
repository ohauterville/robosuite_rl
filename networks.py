import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=128,
        name="critic",
        checkpoint_dir="tmp/td3/run_0",
        learning_rate=10e-3,
    ):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=0.005
        )

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"Created CriticNetwork on device: {self.device}")

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)

        return q1

    def save_checkpoint(self, best_models=False):
        if not best_models:
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), self.checkpoint_file + "_best")

    def load_checkpoint(self, best_models=False):
        if not best_models:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(self.checkpoint_file + "_best"))


class ActorNetwork(nn.Module):

    def __init__(
        self,
        input_dims,
        fc1_dims=256,
        fc2_dims=128,
        learning_rate=10e-3,
        n_actions=2,
        name="actor",
        checkpoint_dir="tmp/td3/run_0",
    ):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"Created ActorNetwork on device: {self.device}")

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = T.tanh(self.output(x))
        return x

    def save_checkpoint(self, best_models=False):
        if not best_models:
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), self.checkpoint_file + "_best")

    def load_checkpoint(self, best_models=False):
        if not best_models:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(self.checkpoint_file + "_best"))
