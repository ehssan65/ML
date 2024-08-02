

import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

import torch
import time
torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
     

# Create CarRacing environment
env = gymnasium.make('CarRacing-v2', render_mode="human")

# Initialize PPO
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=300000, progress_bar=True)

# Save the model
model.save("ppo_car_racing")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
     
