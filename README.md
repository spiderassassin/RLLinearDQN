# RLNonLinearDQN

## Setup

```
pip install flappy-bird-gymnasium
```

Install pytorch (https://pytorch.org/get-started/locally/) and CUDA (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

For example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To use config from yml file:

```
pip install pyyaml
```

## Running

The provided `launch.json` file can be used to debug the program through VS Code. In the Run and Debug tab, just set it as the configuration file, then click the green play button.

Otherwise, train with a config file using the following command:

```
python3 agent.py config_file_name --network-type=linear --train --power
```

Where:
- `--network-type` can be one of [linear, resnet, nonlinear]
- `--train` means the model is being trained, otherwise it runs inferencing with visualization
- without `--power`, by default the program selects the layers to run by using a linearly increasing number of layers
- `--power` means that it instead selects layers that are powers of whichever increment is specified in the config file

The config file should follow this format:

```
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 100000
  mini_batch_size: 32
  epsilon_init: 1
  epsilon_decay: 0.9995
  epsilon_min: 0.05
  alpha: 0.01
  gamma: 0.99
  hidden_dim: 10
  layer_increment: 1
  min_layers: 1
  max_layers: 1
  stop_on_reward: 500
  stop_after_episodes: 200000
  seeds: 5
```

## Acknowledgements

Q Learning implementation: adapted from [Johnny Code's DQN tutorial video series](https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi).
