""""
    An implementation of ResEmoteNet based on the GitHub repo of the paper that 
    introduced it for FER classisfication.
"""
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
        Squeeze-Excitation network that adaptively learns the interdependence 
        between different channles
    """

    def __init__(self, in_ch, r=16):
        """
            in_ch: dimension of input channel
            out_ch: dimension of output channel
            r: reduction ratio
        """
        super().__init__()
        # Define an average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Define a sequential layer that learns the relationship between channels
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get batch_size and channel dimension
        bs, c, _, _ = x.size()
        # Squeeze each channel into a numeric descriptor
        y = self.avg_pool(x).view(bs, c)
        # Pass each descriptor into an excitation layer
        y = self.fc(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
        Residual network to tackle performance degradation
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Define residual function
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)

        # Define identity mapping
        self.shortcut = nn.Sequential()
        # Add optional conv layer if input and output channel don't match
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride ,padding=0),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y += self.shortcut(x)
        return F.relu(y)
    
class ResEmoteNet(nn.Module):
    """
        A neural network architecture consisting of three conv network blocks, one SE block and
        three residual net blocks
    """
    def __init__(self):
        super().__init__()
        # First conv layer followed by batch norm layer 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second conv layer followed by batch norm layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third conv layer followed by batch norm layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # A ReLU activation layer
        self.relu = nn.ReLU(inplace=True)

        # Squeeze-excitation block
        self.se = SEBlock(256)

        # Three residual blocks
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        # A pooling layer for consistent ouput dimension
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers 
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        # Dropout layers to reduce overfitting
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        # Convert to class probability distribution
        self.fc4 = nn.Linear(256, 7)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Apply SE block
        x = self.se(x)

        # Apply residual net
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Apply average pooling
        x = self.pool(x)
        # Reshape previous output to [batch, channel]
        x = x.view(x.size(0), -1)

        # Add fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout2(x)

        x = self.fc4(x)

        return x





        
    