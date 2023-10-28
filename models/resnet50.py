import torch.nn as nn
import torch.nn.functional as F


class BuildingBlock(nn.Module):
    """
    BuildingBlock

    (Bottleneck) Building block explained in the paper Deep Residual Learning for Image Recognition
    This building block need to have input and output channels same sized.
    """
    def __init__(self, extremum_channels, intermediate_channels):
        """
        Parameters
        ----------
        extremum_channels : int
            input and ouptut channel size
        intermediate_channels : int
            intermediate channel size (usually smaller)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(extremum_channels, intermediate_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(intermediate_channels, extremum_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(extremum_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        
        identity = x.clone()

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        x = self.relu3(x)

        return x


class DownSampleBuildingBlock(nn.Module):
    """
    DownSampleBuildingBlock

    This Building block handle down sampling and channels size increase.
    It's the first Building Block of each new stacked block layers.

    """
    def __init__(self, input_channels, output_channels, intermediate_channels, stride):
        """
        Parameters
        ----------
        input_channels : int
            input channel size
        output_channels : int
            ouptut channel size
        intermediate_channels : int
            intermediate channel size
        stride : int
            stride of the second convolution layer and of the projection shortcut.
            When set to 1, the down sampling block don't reduce the size of feature maps (used only
            for the first building block of the first stacked blocks)
            When set to 2, the down sampling block reduce the size of feature maps and the input (with
            a convolution 1x1 with stride 2)

        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(intermediate_channels, output_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu3 = nn.ReLU()
      
        self.proj_shortcut_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        self.bn_proj = nn.BatchNorm2d(output_channels)

    def forward(self, x):

        identity = self.bn_proj(self.proj_shortcut_conv(x.clone()))

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        x = self.relu3(x)

        return x 


class resnet50(nn.Module):
    """
    resnet50 implementation.
    """
    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = num_channels, out_channels = 64, kernel_size = 7, stride = 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.mp1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.stacked_blocks_1 = self.stack_blocks(3, False, 64, 256, 64, 1)
        self.stacked_blocks_2 = self.stack_blocks(4, True, 256, 512, 128, 2)
        self.stacked_blocks_3 = self.stack_blocks(6, True, 512, 1024, 256, 2)
        self.stacked_blocks_4 = self.stack_blocks(3, True, 1024, 2048, 512, 2)

        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = self.stacked_blocks_1(x)
        x = self.stacked_blocks_2(x)
        x = self.stacked_blocks_3(x)
        x = self.stacked_blocks_4(x)
        
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def stack_blocks(self, n_blocks, stack_input_channels, block_extremum_channels, block_intermediate_channels, stride):
        stacked_blocks = []

            
        stacked_blocks.append(DownSampleBuildingBlock(input_channels=stack_input_channels, 
                                                      output_channels=block_extremum_channels, 
                                                      intermediate_channels=block_intermediate_channels,
                                                      stride=stride))

        for block in range(n_blocks - 1):
            stacked_blocks.append(BuildingBlock(extremum_channels=block_extremum_channels, 
                                                intermediate_channels=block_intermediate_channels))
        
        return nn.Sequential(*stacked_blocks)

