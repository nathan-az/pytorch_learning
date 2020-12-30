import gym

import torch

from dqn.config import get_test_config
from dqn.operations import get_screen, get_num_actions, test_model
from models import DQN


env = gym.make("CartPole-v0").unwrapped
env.reset()
config = get_test_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_screen = get_screen(env, config)
_, channels, screen_height, screen_width = init_screen.shape
num_actions = get_num_actions(env)

trained_net = DQN(channels, screen_height, screen_width, num_actions).to(device)
trained_net.load_state_dict(torch.load(config.SAVE_PATH))

episode_durations = []
test_model(
    env=env,
    trained_net=trained_net,
    config=config,
    enable_plot_durations=True,
    episode_durations=episode_durations,
)
