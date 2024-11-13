import torch
from torch import nn
from torch.nn import functional as F

# Implementation of a linear neural network for approximating Q-Learning.
class LinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[]):
        '''
        input_dim: number of dimensions for input to network
        output_dim: number of dimensions for output from network
        layers: list of dims for each hidden layer as int
        '''
        super(LinearNN, self).__init__()

        # layers for model
        self.layers = nn.ParameterList()

        # First layer
        self.layers.append(nn.Linear(input_dim, layers[0]))
        
        # Create however many linear layers supplied as an argument
        if len(layers) > 0:
            for i in range(len(layers) - 1):
                self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
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
    model = LinearNN(4, 2, [256, 256, 128])
    print(model)
    # Test with a random input.
    print(model(torch.randn(1, 4)))
