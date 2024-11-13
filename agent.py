import argparse
import flappy_bird_gymnasium
import gymnasium as gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import yaml
from datetime import datetime, timedelta
from linear import LinearNN
from experience_replay import ReplayMemory
from torch import nn

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
# Save images to file.
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Can also force CPU if that ends up being faster.
# device = 'cpu'

# The RL agent that will interact with the environment.
class Agent:
    def __init__(self, config_set):
        with open('config.yml', 'r') as f:
            self.config = yaml.safe_load(f)[config_set]
        
        self.config_set = config_set
        self.env_id = self.config['env_id']
        self.replay_memory_size = self.config['replay_memory_size']
        self.mini_batch_size = self.config['mini_batch_size']
        self.epsilon_init = self.config['epsilon_init']
        self.epsilon_decay = self.config['epsilon_decay']
        self.epsilon_min = self.config['epsilon_min']
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.hidden_dim = self.config['hidden_dim']
        self.stop_on_reward = self.config['stop_on_reward']
        self.layers = [self.hidden_dim for i in range(self.config['layers'])]
        
        self.loss_function = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{config_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{config_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{config_set}.png")

    def run(self, training=True, render=False):
        if training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Starting training... ({self.config_set})"
            print(log_message)
            with open(self.LOG_FILE, 'w') as f:
                f.write(log_message + '\n')

        # TODO: try with flappy bird if cart pole works!
        # env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        episode_rewards = []

        policy_network = LinearNN(num_states, num_actions, self.layers).to(device)

        if training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init
            epsilon_history = []
            best_reward = -9999999

            # Initialize the optimizer.
            self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=self.alpha)
        else:
            # Load the model from file.
            policy_network.load_state_dict(torch.load(self.MODEL_FILE))
            policy_network.eval()

        for episode in itertools.count():
            s, _ = env.reset()
            # Convert to tensor for pytorch.
            s = torch.tensor(s, dtype=torch.float, device=device)

            term = False
            episode_reward = 0

            while not term and episode_reward < self.stop_on_reward:
                # Epsilon greedy action selection.
                if training and random.random() < epsilon:
                    a = env.action_space.sample()
                    # Convert to tensor for pytorch.
                    a = torch.tensor(a, dtype=torch.long, device=device)
                else:
                    # Save time, calculates gradient automatically, but don't need it for evaluation.
                    with torch.no_grad():
                        # Argmax index corresponds to the optimal action.
                        a = policy_network(s.unsqueeze(dim=0)).squeeze().argmax()

                # Take the action.
                s_next, r, term, _, _ = env.step(a.item())
                episode_reward += r
                # Convert to tensor for pytorch.
                s_next = torch.tensor(s_next, dtype=torch.float, device=device)
                r = torch.tensor(r, dtype=torch.float, device=device)

                if training:
                    memory.append(s, a, r, s_next, term)

                s = s_next

            episode_rewards.append(episode_reward)

            if training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward:0.1f} ({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')

                    torch.save(policy_network.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every so often.
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(episode_rewards, epsilon_history)
                    last_graph_update_time = current_time

                # Once we have enough experience.
                if len(memory) > self.mini_batch_size:
                    # Sample a mini batch.
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_network)

                    # Decay epsilon.
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                        
                # stop training
                if episode_reward >= self.stop_on_reward:
                    return

    # Calculate the target value and train the policy.
    def optimize(self, mini_batch, policy_network):
        # Extract each component of the entire mini batch.
        s, a, r, s_next, term = zip(*mini_batch)

        # Convert to tensors.
        s = torch.stack(s)
        a = torch.stack(a)
        s_next = torch.stack(s_next)
        r = torch.stack(r)
        # term is a boolean, for simplicity, convert to 1 or 0.
        term = torch.tensor(term).float().to(device)

        with torch.no_grad():
            target = r + (1 - term) * self.gamma * policy_network(s_next).max(dim=1)[0]
        
        prediction = policy_network(s).gather(dim=1, index=a.unsqueeze(dim=1)).squeeze()
        
        # Calculate the loss.
        loss = self.loss_function(prediction, target)

        # Backpropagate the loss (optimize the model).
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, episode_rewards, epsilon_history):
        fig = plt.figure(1)

        # Plot average rewards (y) against episodes (x).
        avg_rewards = np.zeros(len(episode_rewards))
        for i in range(len(avg_rewards)):
            avg_rewards[i] = np.mean(episode_rewards[max(0, i-99):(i+1)])
        # Plot on a 1 row x 2 column grid, at cell 1.
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Average reward')
        plt.plot(avg_rewards)

        # Plot epsilon decay (y) against episodes (x).
        plt.subplot(122)
        plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train or test a DQN agent.')
    parser.add_argument('config', help='Name of config set in config.yml.')
    parser.add_argument('--train', action='store_true', help='Train the agent.')
    parser.add_argument('--all', action='store_true', help='Run full training suite.')
    args = parser.parse_args()

    # run all training configs
    if args.all:
        with open(f'{args.config}.yml', 'r') as f:
            config = yaml.safe_load(f)
            
            for param_set in config:
                
                agent = Agent(config_set=param_set)
                agent.run(training=True)
                
    else:
        

        agent = Agent(config_set=args.config)

        if args.train:
            agent.run(training=True)
        else:
            agent.run(training=False, render=True)
