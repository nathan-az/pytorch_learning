from typing import Literal

from dacite import from_dict
from dataclasses import dataclass

_train_config = dict(
    BATCH_SIZE=128,
    GAMMA=0.999,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=200,
    NUM_EPISODES=300,
    TARGET_UPDPATE=10,
    GRAYSCALE=False,
    TRIM_WIDTH=True,
    DECAY_BY="step",
    SAVE_NAME="cartpole_model",
    SAVE_BEST_MODEL=True,
)

_test_config = dict(
    BATCH_SIZE=128,
    GAMMA=0.999,
    EPS_START=0.0,
    EPS_END=0.0,
    EPS_DECAY=200,
    NUM_EPISODES=300,
    TARGET_UPDPATE=100,
    GRAYSCALE=False,
    TRIM_WIDTH=True,
    DECAY_BY="step",
    MODEL_PATH="original_300ep_best.pth",
)


@dataclass
class TrainConfig:
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
    SAVE_NAME: str
    SAVE_BEST_MODEL: bool


@dataclass
class TestConfig:
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
    MODEL_PATH: str


def get_train_config():
    return from_dict(TrainConfig, _train_config)


def get_test_config():
    return from_dict(TestConfig, _test_config)
