from dacite import from_dict
from dataclasses import dataclass

_cfg = dict(
    BATCH_SIZE=128,
    GAMMA=0.999,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=200,
    TARGET_UPDPATE=10,
)


@dataclass
class Config:
    BATCH_SIZE: int
    GAMMA: float
    EPS_START: float
    EPS_END: float
    EPS_DECAY: int
    TARGET_UPDPATE: int


def get_config():
    return from_dict(Config, _cfg)
