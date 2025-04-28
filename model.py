from torch import nn


class MLP(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(MLP, self).__init__()
        self.hparams = hparams
        self.device = device

        nodes = hparams["nodes"]
        layers = hparams["layers"]
        input_size = 1
        output_size = 1

        net = [nn.Linear(input_size, nodes), nn.GELU()]
        for _ in range(layers - 1):
            net.append(nn.Linear(nodes, nodes))
            net.append(nn.GELU())
        net.append(nn.Linear(nodes, output_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    default_hparams = {
        "num_conv_layers": 3,
        "num_fc_layers": 2,
        "conv_channels": 16,
        "fc_nodes": 128,
    }
    
    def __init__(self, hparams, device="cpu"):
        super(CNN, self).__init__()
        self.hparams = hparams
        self.device = device
        
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        # Extract hyperparameters with defaults
        num_conv_layers = int(hparams.get("num_conv_layers", self.default_hparams["num_conv_layers"]))
        num_fc_layers = int(hparams.get("num_fc_layers", self.default_hparams["num_fc_layers"]))
        conv_channels = int(hparams.get("conv_channels", self.default_hparams["conv_channels"]))
        fc_nodes = int(hparams.get("fc_nodes", self.default_hparams["fc_nodes"]))
        
        # Convolutional layers
        in_channels = 1  # MNIST images are grayscale
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = conv_channels
        
        # Fully connected layers
        fc_input = conv_channels * (28 // (2 ** num_conv_layers)) ** 2
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_input, fc_nodes))
            self.fc_layers.append(nn.ReLU())
            fc_input = fc_nodes
        
        self.fc_layers.append(nn.Linear(fc_nodes, 10))  # 10 classes for MNIST
    
    def forward(self, x):
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
