import argparse
import gymnasium as gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import torch
import yaml
from experience_replay import ReplayMemory
from datetime import datetime, timedelta
from nonlinear import NonLinearNN
from linear_copy import LinearNN
from linear_resnet import LinearResNetNN
from linear_norm import LinearNormNN
from linear_res_no_norm import LinearResNN
from torch import nn
import torchprofile

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
    def __init__(self, config_set, config_file, layers, network_type='linear'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)[config_set]

        self.config_set = config_set
        self.config_file = config_file
        self.env_id = self.config['env_id']
        self.replay_memory_size = self.config['replay_memory_size']
        self.mini_batch_size = self.config['mini_batch_size']
        self.epsilon_init = self.config['epsilon_init']
        self.epsilon_decay = self.config['epsilon_decay']
        self.epsilon_min = self.config['epsilon_min']
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.hidden_dim = self.config['hidden_dim']
        self.layer_increment = self.config['layer_increment']
        self.min_layers = self.config['min_layers']
        self.max_layers = self.config['max_layers']
        self.stop_on_reward = self.config['stop_on_reward']
        self.stop_after_episodes = self.config['stop_after_episodes']
        self.seeds = self.config['seeds']
        self.num_flops = 0
        
        # Try grabbing "block" type from config. If it doesn't have it, proceed
        # as normal
        try:
            self.block = self.config['block']
            
            # Because basic blocks have 2 layers, halve the number of layers
            # Resnet implementation generates as many blocks as layers supplied
            # need to subtract 2 from num layers to account for input and output layers
            if self.block == "Basic":
                self.layers = [self.hidden_dim for _ in range(max(1, (layers - 2)//2))]
                
            # same as basic block but bottleneck uses 3 layers instead
            elif self.block == "Bottleneck":
                self.layers = [self.hidden_dim for _ in range(max(1, (layers - 2)//3))]
                
            else:
                self.layers = [self.hidden_dim for _ in range(layers)]

        except:
            self.block = "Basic"
            self.layers = [self.hidden_dim for _ in range(layers)]
            
        self.network_type = network_type
        self.loss_function = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{config_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{config_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{config_set}.png")

    # Log a message to the console and the log files.
    def log(self, message):
        timestamp = datetime.now().strftime(DATE_FORMAT)
        print(f"{timestamp}: {message}")
        with open(self.LOG_FILE, 'a') as f:
            f.write(f"{timestamp}: {message}\n")
        with open(AGGREGATE_FILE, 'a') as f:
            f.write(f"{timestamp}: {message}\n")

    # Run a specific configuration with multiple seeds, returning the average number of timesteps it took to reach the max reward.
    def run(self, training=True, render=False):
        durations = []
        total_layer_passes = []
        all_flops = []
        for _ in range(self.seeds):
            # Generate random seed.
            seed = random.randint(0, 100000)
            self.log(f"Seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            # Run the agent with the given seed.
            duration, layer_passes, num_flops = self.run_single(seed, training, render)
            durations.append(duration)
            total_layer_passes.append(layer_passes)
            all_flops.append(num_flops)

        avg_duration = np.mean(durations)
        avg_layer_passes = np.mean(total_layer_passes)
        avg_flops = np.mean(all_flops)
        self.log(f"Average duration: {avg_duration:0.1f} timesteps, Average layer passes: {avg_layer_passes:0.1f}, Average number of FLOPs: {avg_flops:0.1f} ({self.config_set})")

        return avg_duration, avg_layer_passes, avg_flops

    # Runs with a single seed, returning the number of timesteps it took to reach the max reward.
    def run_single(self, seed, training=True, render=False):
        if training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            self.log(f"Starting training with {len(self.layers)} layer(s)... ({self.config_set})")

        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        timestep = 0
        layer_passes = 0
        episode_rewards = []

        if self.network_type == 'linear':
            policy_network = LinearNN(num_states, num_actions, self.layers).to(device)
        elif self.network_type == 'resnet':
            policy_network = LinearResNetNN(num_states, num_actions, self.layers, block=self.block).to(device)
        elif self.network_type == 'res_only':
            policy_network = LinearResNN(num_states, num_actions, self.layers).to(device)
        elif self.network_type == 'norm_only':
            policy_network = LinearNormNN(num_states, num_actions, self.layers).to(device)
        else:
            # policy_network = NonLinearNN(num_states, num_actions, self.layers).to(device)
            policy_network = LinearResNetNN(num_states, num_actions, self.layers, block=self.block, nonlinear=True).to(device)

        num_flops = 0

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
            # Set the seed at the start just once.
            env_seed = seed if episode == 0 else None
            s, _ = env.reset(seed=env_seed)
            # Convert to tensor for pytorch.
            s = torch.tensor(s, dtype=torch.float, device=device)

            term = False
            trunc = False
            episode_reward = 0

            # Episode end is either environment terminates (failed) or environment truncates (solved).
            while not (term or trunc):
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
                        num_flops += (torchprofile.profile_macs(policy_network, s.unsqueeze(dim=0)) * 2)
                        layer_passes += len(self.layers)

                # Take the action.
                s_next, r, term, trunc, _ = env.step(a.item())
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
                    self.log(f"New best reward: {episode_reward:0.1f} ({(episode_reward - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model...")
                    torch.save(policy_network.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every so often.
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(episode_rewards, epsilon_history, seed)
        
                    last_graph_update_time = current_time

                # Once we have enough experience.
                if len(memory) > self.mini_batch_size:
                    # Sample a mini batch.
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_network)

                    # Decay epsilon.
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                # Stop training if reached an average of target reward over the last 100.
                # IMPROMPTU MODIFICATION: stop once max reward recieved once
                if episode_rewards[-1] >= self.stop_on_reward:
                    self.log(f"Reached target after {episode} episodes with number of timesteps: {timestep}, layer passes: {layer_passes}, and number of FLOPs: {num_flops}.")
                    return timestep, layer_passes, num_flops
                # Or if it's been too long.
                if episode > self.stop_after_episodes:
                    self.log(f"Giving up after {episode} episodes with number of timesteps: {timestep}, layer passes: {layer_passes}, and number of FLOPs: {num_flops}.")
                    return timestep, layer_passes, num_flops

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

    # Save the graph of the average reward and epsilon decay.
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
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(f'{self.GRAPH_FILE[:-4]}_{seed}.png')
        plt.close(fig)

    # Save the graph of the rate of change in average reward.
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

        # Remove .yml file extension from file name.
        file_name = self.config_file.split('.')[0]
        if not os.path.exists(f"./runs/{file_name}/{self.config_set}"):
            os.makedirs(f"./runs/{file_name}/{self.config_set}")

        fig.savefig(f"./runs/{file_name}/{self.config_set}/{space}_rate_{seed}.png")
        plt.close(fig)


# Save the aggregate graph of the number of layers vs. the average duration taken to reach the max reward.
def save_aggregate_graph(data, filename, xLabel, yLabel):
    fig = plt.figure(1)
    keys = data.keys()
    values = data.values()
    x = range(len(keys))
    # Create a bar chart of the data, with the layers as categories.
    plt.bar(x, values, color='blue') 
        
    plt.xticks(x, keys)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    fig.savefig(os.path.join(RUNS_DIR, filename))
    plt.close(fig)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train or test a Q Learning agent.')
    parser.add_argument('config', help="Name of the config file.")
    parser.add_argument('--network-type', choices=['linear', 'resnet', 'nonlinear', 'res_only', 'norm_only'], help="Type of neural network to use.")
    parser.add_argument('--train', action='store_true', help='Train the agent.')
    parser.add_argument('--power', action='store_true', help='Run config set for exponential intervals of layers (default is linear).')
    args = parser.parse_args()


    # Don't remove the runs directory if you want to save multiple results in a row
    # # Clean out the runs directory.
    # for file in os.listdir(RUNS_DIR):
    #     # By default remove files.
    #     try:
    #         os.remove(os.path.join(RUNS_DIR, file))
    #     # Remove directories if present (don't need to be empty).
    #     except:
    #         shutil.rmtree(os.path.join(RUNS_DIR, file))

    # Copy the config file to the runs directory.
    shutil.copy(f'{args.config}.yml', RUNS_DIR)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(f'{args.config}.yml', 'r') as f:
        config = yaml.safe_load(f)

        # Track the number of layers and resulting duration taken to reach the max reward.
        avg_durations = {}
        # Do similar for the number of layers that data is passed through.
        avg_layer_passes = {}
        # Do again but for FLOPs
        num_flops = {}

        # Local function to run an agent with a specific number of layers.
        def run_agent(param_set, i):
            agent = Agent(config_set=param_set, config_file=f'{args.config}.yml', layers=i, network_type=args.network_type)
            avg_duration, avg_layer_pass, avg_flops = agent.run(training=args.train, render=not args.train)
            avg_durations[i] = avg_duration
            avg_layer_passes[i] = avg_layer_pass
            num_flops[i] = avg_flops
            save_aggregate_graph(avg_durations, "aggregate_timesteps.png", "Size of Network (Layers)", "Performance (Timesteps)")
            save_aggregate_graph(avg_layer_passes, "aggregate_layers.png", "Size of Network (Layers)", "Energy (Total Number of Layers Used During Learning)")
            save_aggregate_graph(num_flops, f"flops_per_layer_{agent.config_set}.png", "Size of Network (Layers)", "Number of FLOPs")

        for param_set in config:
            layer_increment = config[param_set]['layer_increment']
            min_layers = config[param_set]['min_layers']
            max_layers = config[param_set]['max_layers']
            # Exponential.
            if args.power:
                exponent = 0
                while (i := layer_increment**exponent) <= max_layers:
                    if i >= min_layers:
                        run_agent(param_set, i)
                    exponent += 1
            # Linear.
            else:
                for i in range(min_layers, max_layers + 1, layer_increment):
                    run_agent(param_set, i)
