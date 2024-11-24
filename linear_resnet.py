import torch
from torch import nn

# Residual block for an adaptation of ResNet to our linear network.
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        # Skipping the typical ReLU activation here to stay linear.
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.downsample = None

        # Downsample only if the input and output dimensions are different.
        if input_dim != output_dim:
            self.downsample = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(self, x):
        # Keep everything linear here as well.
        identity = x
        x = self.ln1(self.fc1(x))
        x = self.ln2(self.fc2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        # Skip connection.
        x += identity

        return x

# Implementation of a linear neural network for approximating Q-Learning using a ResNet-inspired approach.
class LinearResNetNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[]):
        '''
        input_dim: number of dimensions for input to network
        output_dim: number of dimensions for output from network
        layers: list of dims for each hidden layer as int
        '''
        super(LinearResNetNN, self).__init__()

        # layers for model
        self.layers = nn.ParameterList()

        # First layer
        self.layers.append(ResidualBlock(input_dim, layers[0]))
        
        # Create however many linear layers supplied as an argument
        if len(layers) > 0:
            for i in range(len(layers) - 1):
                self.layers.append(ResidualBlock(layers[i], layers[i+1]))
        
        # And the output layer.
        self.layers.append(nn.Linear(layers[-1], output_dim))

    def forward(self, x):
        result = x
        # feedforward the result into each layer
        for layer in self.layers:
            result = layer(result)
            
        return result
        

if __name__ == '__main__':
    # Test the model.
    model = LinearResNetNN(4, 2, [256, 256, 128])
    print(model)
    # Test with a random input.
    print(model(torch.randn(1, 4)))
