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

Otherwise, train using the following command:

```
python3 agent.py cartpole1 --train
```

And to test with visualization:

```
python3 agent.py cartpole1
```

## Acknowledgements

DQN implementation: slightly adapted from [Johnny Code's DQN tutorial video series](https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi).
