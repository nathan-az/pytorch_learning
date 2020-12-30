import gym

import torch
import torch.optim as optim

from dqn.config import get_train_config
from dqn.operations import get_screen, get_num_actions, train_model
from models import DQN, ReplayMemory


env = gym.make("CartPole-v0").unwrapped
env.reset()
config = get_train_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_screen = get_screen(env, config)
_, channels, screen_height, screen_width = init_screen.shape
num_actions = get_num_actions(env)


policy_net = DQN(channels, screen_height, screen_width, num_actions).to(device)
target_net = DQN(channels, screen_height, screen_width, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimiser = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

episode_durations = []
train_model(
    env=env,
    policy_net=policy_net,
    target_net=target_net,
    optimiser=optimiser,
    save_model=True,
    memory=memory,
    config=config,
    enable_plot_durations=True,
)
