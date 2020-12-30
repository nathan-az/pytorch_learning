from typing import Literal

from dacite import from_dict
from dataclasses import dataclass

_train_config = dict(
    BATCH_SIZE=128,
    GAMMA=0.999,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=200,
    NUM_EPISODES=500,
    TARGET_UPDPATE=10,
    GRAYSCALE=True,
    TRIM_WIDTH=True,
    DECAY_BY="step",
    SAVE_PATH="cartpole_model.pth",
)

_test_config = dict(
    BATCH_SIZE=128,
    GAMMA=0.999,
    EPS_START=0,
    EPS_END=0,
    EPS_DECAY=0,
    NUM_EPISODES=100,
    TARGET_UPDPATE=100,
    GRAYSCALE=True,
    TRIM_WIDTH=True,
    DECAY_BY="step",
    SAVE_PATH="cartpole_model.pth",
)


@dataclass
class Config:
    BATCH_SIZE: int
    GAMMA: float
    EPS_START: float
    EPS_END: float
    EPS_DECAY: int
    NUM_EPISODES: int
    TARGET_UPDPATE: int
    GRAYSCALE: bool
    TRIM_WIDTH: bool
    DECAY_BY: Literal["episode", "step"]
    SAVE_PATH: str


def get_train_config():
    return from_dict(Config, _train_config)


def get_test_config():
    return from_dict(Config, _test_config)
