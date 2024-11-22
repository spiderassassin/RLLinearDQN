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
AGGREGATE_FILE = os.path.join(RUNS_DIR, "aggregate.log")
# Save images to file.
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#if(torch.backends.mps.is_available()):
    #device = 'mps'
# Can also force CPU if that ends up being faster.
# device = 'cpu'

# The RL agent that will interact with the environment.
class Agent:
    def __init__(self, config_set, config_name):
        #HAVE TO CHANGE THE NAME OF THE YAML FILE
        with open(config_name, 'r') as f:
            self.config = yaml.safe_load(f)[config_set]
        
        self.config_name = config_name
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
        self.stop_after_episodes = self.config['stop_after_episodes']
        self.layers = [self.hidden_dim for i in range(self.config['layers'])]
        self.seeds = self.config['seeds']
        
        self.loss_function = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{config_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{config_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{config_set}.png")

    # RUn a specific configuration with multiple seeds, returning the average number of timesteps it took to reach the max reward.
    def run(self, training=True, render=False):
        durations = []
        total_layer_passes = []
        for _ in range(self.seeds):
            seed = random.randint(0, 100000)
            log_message = f"Seed: {seed}"
            print(log_message)
            with open(self.LOG_FILE, 'a') as f:
                f.write(log_message + '\n')
            with open(AGGREGATE_FILE, 'a') as f:
                f.write(log_message + '\n')
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            duration, layer_passes = self.run_single(seed, training, render)
            durations.append(duration)
            total_layer_passes.append(layer_passes)

        avg_duration = np.mean(durations)
        avg_layer_passes = np.mean(total_layer_passes)
        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Average duration: {avg_duration:0.1f} timesteps, Average layer passes: {avg_layer_passes:0.1f} ({self.config_set})"
        print(log_message)
        with open(self.LOG_FILE, 'a') as f:
            f.write(log_message + '\n')
        with open(AGGREGATE_FILE, 'a') as f:
            f.write(log_message + '\n')
        
        return avg_duration, avg_layer_passes

    # Runs with a single seed, returning the number of timesteps it took to reach the max reward.
    def run_single(self, seed, training=True, render=False):
        if training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Starting training... ({self.config_set})"
            print(log_message)
            with open(self.LOG_FILE, 'w') as f:
                f.write(log_message + '\n')
            with open(AGGREGATE_FILE, 'a') as f:
                f.write(log_message + '\n')

        # TODO: try with flappy bird if cart pole works!
        #env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        timestep = 0
        layer_passes = 0
        episode_rewards = []

        policy_network = LinearNN(num_states, num_actions, self.layers).to(device)

        if training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init
            epsilon_history = []
            best_reward = -1

            # Initialize the optimizer.
            self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=self.alpha)
        else:
            # Load the model from file.
            policy_network.load_state_dict(torch.load(self.MODEL_FILE))
            policy_network.eval()

        for episode in itertools.count():
            env_seed = seed if episode == 0 else None
            s,_ = env.reset(seed=env_seed)
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
                        layer_passes += len(self.layers)

                # Take the action.
                s_next, r, term, _, _ = env.step(a.item())
                episode_reward += r
                # Convert to tensor for pytorch.
                s_next = torch.tensor(s_next, dtype=torch.float, device=device)
                r = torch.tensor(r, dtype=torch.float, device=device)

                if training:
                    memory.append(s, a, r, s_next, term)
                    pass

                s = s_next
                timestep += 1

            episode_rewards.append(episode_reward)

            if training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward:0.1f} ({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')
                    with open(AGGREGATE_FILE, 'a') as f:
                        f.write(log_message + '\n')

                    torch.save(policy_network.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every so often.
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(episode_rewards, epsilon_history, seed)
                    self.save_rate_graph(episode_rewards, space=1, seed=seed)
                    self.save_rate_graph(episode_rewards, space=50, seed=seed)
                    self.save_rate_graph(episode_rewards, space=100, seed=seed)
                    self.save_rate_graph(episode_rewards, space=200, seed=seed)
                    self.save_rate_graph(episode_rewards, space=300, seed=seed)
                    self.save_rate_graph(episode_rewards, space=400, seed=seed)
                    self.save_rate_graph(episode_rewards, space=500, seed=seed)

                    last_graph_update_time = current_time

                # Once we have enough experience.
                if len(memory) > self.mini_batch_size:
                    # Sample a mini batch.
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_network)

                    # Decay epsilon.
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                        
                # Stop training if reached the target reward.
                if episode_reward >= self.stop_on_reward:
                    return timestep, layer_passes
                # Or if it's been too long.
                if episode > self.stop_after_episodes:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Giving up after {episode} episodes."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')
                    with open(AGGREGATE_FILE, 'a') as f:
                        f.write(log_message + '\n')
                    return timestep, layer_passes

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

    def save_graph(self, episode_rewards, epsilon_history, seed=None):
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

        fig.savefig(f'{self.GRAPH_FILE[:-4]}_{seed}.png')
        plt.close(fig)

    def save_rate_graph(self, episode_rewards, space=1, seed=None):
        
        if len(episode_rewards) < space:
            return
        
        fig = plt.figure(1)

        avg_rewards = np.zeros(len(episode_rewards))
        for i in range(len(avg_rewards)):
            avg_rewards[i] = np.mean(episode_rewards[max(0, i-99):(i+1)])
        # Plot on a 1 row x 2 column grid, at cell 1.
        plt.subplot(111)
        plt.xlabel(f"Number of Episodes (x{space})")
        plt.ylabel('Change in Average Reward')
        
        grad = np.gradient(avg_rewards[::space], space)
        plt.plot(grad)
        
        avg_avg = np.zeros(len(grad))
        for i in range(10,len(grad)):
            avg_avg[i] = np.mean(grad[max(0, i-9):(i+1)])
        plt.plot(avg_avg)
        
        # Plot epsilon decay (y) against episodes (x).

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        if not os.path.exists(f"./runs/{self.config_name}/{self.config_set}"):
            os.makedirs(f"./runs/{self.config_name}/{self.config_set}")

        fig.savefig(f"./runs/{self.config_name}/{self.config_set}/{space}_rate_{seed}.png")
        plt.close(fig)

# Save the aggregate graph of the number of layers vs. the average duration taken to reach the max reward.
def save_aggregate_graph(data, filename):
    fig = plt.figure(1)
    keys = data.keys()
    values = data.values()
    x = range(len(keys))
    # Create a bar chart of the data, with the layers as categories.
    plt.bar(x, values, color='blue')
    plt.xticks(x, keys)

    plt.xlabel('Number of Layers')
    plt.ylabel('Average Duration to Reach Max Reward')
    fig.savefig(os.path.join(RUNS_DIR, filename))
    plt.close(fig)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train or test a DQN agent.')
    parser.add_argument('file', help="Name of the config file.")
    parser.add_argument('config', help='Name of config set in config file.')
    parser.add_argument('--train', action='store_true', help='Train the agent.')
    parser.add_argument('--all', action='store_true', help='Run full training suite.')
    args = parser.parse_args()

    # Clean out the runs directory.
    for file in os.listdir(RUNS_DIR):
        os.remove(os.path.join(RUNS_DIR, file))

    # run all training configs
    if args.all:
        with open(f'{args.file}.yml', 'r') as f:
            config = yaml.safe_load(f)

            # Track the number of layers and resulting duration taken to reach the max reward.
            avg_durations = {}
            # Do similar for the number of layers that data is passed through.
            avg_layer_passes = {}
            
            for param_set in config:

                print(param_set)
                agent = Agent(config_set=param_set, config_name=f'{args.config}.yml')
                avg_duration, avg_layer_pass = agent.run(training=True)
                avg_durations[config[param_set]['layers']] = avg_duration
                avg_layer_passes[config[param_set]['layers']] = avg_layer_pass
                save_aggregate_graph(avg_durations, "aggregate_timesteps.png")
                save_aggregate_graph(avg_layer_passes, "aggregate_layers.png")

                
    else:
        agent = Agent(config_set=args.config, config_name=f'{args.file}.yml')

        if args.train:
            agent.run(training=True)
        else:
            agent.run(training=False, render=True)
