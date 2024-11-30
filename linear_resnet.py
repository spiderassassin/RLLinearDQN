import torch
from torch import nn

# Residual block for an adaptation of ResNet to our linear network.
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinear=False):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.downsample = None
        self.nonlinear = nonlinear

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
        
        # if doing non-linear resnet
        if self.nonlinear:
            x = self.relu(x)
            
        x = self.ln2(self.fc2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        # Skip connection.
        x += identity
        
        # if doing non-linear resnet
        if self.nonlinear:
            x = self.relu(x)

        return x
    
'''linear adaptation of resnet bottleneck block. Difference between this and
residual block (referred to as basic block in the resnet paper) is there are
3 layers in each block, not 2. Because the bottleneck layer in original resnet
is using convolutions, we adapt it here for fully connected layers but by dividing
the hidden layer dimension by 4 for the first 2 layers, and then bring it back up
to the full size for the third layer.'''
class BottleneckBlock(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinear=False):
        super(BottleneckBlock, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim//4)
        self.ln1 = nn.LayerNorm(input_dim//4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.lin2 = nn.Linear(input_dim//4, input_dim//4)
        self.ln2 = nn.LayerNorm(input_dim//4)
        
        self.lin3 = nn.Linear(input_dim//4, output_dim)
        self.ln3 = nn.LayerNorm(output_dim)
        
        self.downsample = None
        self.nonlinear = nonlinear

        # Downsample only if the input and output dimensions are different.
        if input_dim != output_dim:
            self.downsample = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(self, x):
        # Keep everything linear here as well.
        identity = x

        x = self.lin1(x)
        x = self.ln1(x)
        
        if self.nonlinear:
            x = self.relu(x)

        x = self.lin2(x)
        x = self.ln2(x)
        
        if self.nonlinear:
            x = self.relu(x)

        x = self.lin3(x)
        x = self.ln3(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        
        if self.nonlinear:
            x = self.relu(x)

        return x

# Implementation of a linear neural network for approximating Q-Learning using a ResNet-inspired approach.
class LinearResNetNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[], block="Basic", nonlinear=False):
        '''
        input_dim: number of dimensions for input to network
        output_dim: number of dimensions for output from network
        layers: list of dims for each hidden layer as int
        '''
        super(LinearResNetNN, self).__init__()

        # layers for model
        self.layers = nn.ParameterList()
        
        if block == "Basic":
            self.block = ResidualBlock
            
        else:
            self.block = BottleneckBlock

        # First layer
        self.layers.append(nn.Linear(input_dim, layers[0]))
        
        # Create however many linear layers supplied as an argument
        if len(layers) > 0:
            for i in range(len(layers) - 1):
                self.layers.append(self.block(layers[i], layers[i+1], nonlinear))
        
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
    model = LinearResNetNN(4, 2, [256, 256, 256], "Bottleneck", True)
    print(model)
    # Test with a random input.
    print(model(torch.randn(1, 4)))
