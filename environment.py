#flappy bird
import flappy_bird_gymnasium
import gymnasium as gym

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()

#cartpole
env = gym.make("CartPole-v1", render_mode="rgb_array")
