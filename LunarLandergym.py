import gymnasium as gym

from stable_baselines3 import DQN


env = gym.make("LunarLander-v2", render_mode="human")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqnlunarlander")

del model # remove to demonstrate saving and loading

model = DQN.load("dqnlunarlander")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()