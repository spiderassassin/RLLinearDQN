import torch
from torch import nn
from torch.nn import functional as F

# Implementation of a deep Q-network.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQN, self).__init__()

        # One hidden layer.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # And the output layer.
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Use ReLU activation (nonlinear) for the hidden layer.
        # TODO: Change this to be linear!
        x = F.relu(self.fc1(x))
        return self.output(x)

if __name__ == '__main__':
    # Test the model.
    model = DQN(4, 2)
    print(model)
    # Test with a random input.
    print(model(torch.randn(1, 4)))
