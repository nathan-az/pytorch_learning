import random
from collections import deque

import numpy as np
import torch

from marionet.models import MarioNet


class Mario:
    def __init__(
        self, state_dim, action_dim, save_dir, memory_capacity=100000, batch_size=32
    ):
        # INPUT
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # SETTINGS
        self.use_cuda = torch.cuda.is_available()
        self.net: MarioNet = MarioNet(self.state_dim, self.action_dim).float()
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        # ACTIONS
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Q LEARNING PARAMS
        self.gamma = 0.9
        self.burnin = 1e4  # min experiences before training
        self.learn_every = 3  # avoids training on every frame
        self.sync_every = 1e4  # delay between sync of online and target models

        # LEARNING
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.save_every = 5e5

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        else:
            state = torch.tensor(state.__array__())
            if self.use_cuda:
                state = state.cuda()
            state = state.unsqueeze(0)
            action_values = self.net(state, model_type="online")
            action_idx = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)
        self.curr_step += 1
        return action_idx

    def cache(self, state, action, reward, next_state, done):
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        if self.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action.squeeze, reward.squeeze, next_state, done

    def td_estimate(self, state, action):
        # below notation means run the net on the state vector, get the # corresponding ot batch_size
        # and corresponding to the specific action
        current_Q = self.net(state, model_type="online")[
            np.arange(self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model_type="online")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model_type="target")[
            np.arange(0, self.batch_size), best_action
        ]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"marionet_{int(self.curr_step % self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict, exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, action, next_state, reward, done = self.recall()
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_estimate, td_target)
        return (td_estimate.mean().item(), loss)
