"""RL algorithms."""
from pyro.algos.dqn import DQN
from pyro.algos.rem import REM
from pyro.algos.sac import SAC
# from pyro.algos.vpg import VPG
# from pyro.algos.trpo import TRPO

__all__ = [
    'DQN',
    'REM',
    'SAC',
    # 'VPG',
    # 'TRPO',
]

