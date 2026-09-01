# envs/__init__.py
from gymnasium.envs.registration import register
from .env_1uav_5user_power import UAV6GMultiUserEnv

register(
    id='UAV-v0',
    entry_point='envs.env_1uav_5user_power:UAV6GMultiUserEnv',
    max_episode_steps=200,
)