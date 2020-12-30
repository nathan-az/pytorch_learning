import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

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
