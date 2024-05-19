class BrainNetworkModel(nn.Module):
    def __init__(self, num_traits, num_classes):
        super(BrainNetworkModel, self).__init__()
        # Convolutional branch
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)  # Input is 1 channel, output is 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected branch for traits
        self.fc1_traits = nn.Linear(num_traits, 50)
        
        # Combined fully connected layers
        self.fc2_combined = nn.Linear(32 + 50, 100)  # 32 from CNN, 50 from traits
        self.fc3_combined = nn.Linear(100, num_classes)

    def forward(self, x_network, x_traits):
        # Convolutional branch
        x_network = self.pool(F.relu(self.conv1(x_network)))
        x_network = F.relu(self.conv2(x_network))
        x_network = torch.flatten(x_network, 1)  # Flatten all dimensions except batch
        
        # Traits branch
        x_traits = F.relu(self.fc1_traits(x_traits))
        
        # Combine the outputs from the two branches
        x_combined = torch.cat((x_network, x_traits), dim=1)
        
        # Further processing
        x_combined = F.relu(self.fc2_combined(x_combined))
        x_combined = self.fc3_combined(x_combined)
        return x_combined
