import torch
from pathlib import Path
import datetime

from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from marionet.agents import Mario
from marionet.logger import MetricLogger
from marionet.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"]])
env.reset()
next_state, reward, done, info = env.step(action=0)


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 4000
for ep in range(episodes):
    state = env.reset()
    while True:
        action = mario.act(state)
        next_state, reward, done, info = env.step(done)
        mario.cache(state, action, reward, next_state, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done or info["flag_get"]:
            break
    logger.log_episode()
    if ep % 20 == 0:
        logger.record(episode=ep, epsilon=mario.exploration_rate, step=mario.curr_step)
