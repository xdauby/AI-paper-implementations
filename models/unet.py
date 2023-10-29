import torch.nn as nn
 
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mp = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, downsample=True):

        if downsample:
          x = self.mp(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.relu2 = nn.ReLU()

    def forward(self, x1, x2):
        
        x1 = self.upconv(x1)

        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x2 = x2[:,:, diffY//2 : x2.shape[2]- diffY//2, diffX//2 : x2.shape[3]- diffX//2]
        x = torch.cat([x2, x1], dim=1)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNet(nn.Module):
  def __init__(self, num_channels, num_classes):
        super().__init__()

        self.ds1 = DownSamplingBlock(num_channels, 64)
        self.ds2 = DownSamplingBlock(64, 128)
        self.ds3 = DownSamplingBlock(128, 256)
        self.ds4 = DownSamplingBlock(256, 512)
        self.ds5 = DownSamplingBlock(512, 1024)

        self.us1 = UpSamplingBlock(1024)
        self.us2 = UpSamplingBlock(512)
        self.us3 = UpSamplingBlock(256)
        self.us4 = UpSamplingBlock(128)

        self.conv11 = nn.Conv2d(64, num_classes, 1)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.relu2 = nn.ReLU()


  def forward(self, x):

      x1 = self.ds1(x, False)
      x2 = self.ds2(x1)
      x3 = self.ds3(x2)
      x4 = self.ds4(x3)
      x5 = self.ds5(x4)
      
      x6 = self.us1(x5, x4)
      x7 = self.us2(x6, x3)
      x8 = self.us3(x7, x2)
      x9 = self.us4(x8, x1)
      
      x = self.relu2(self.bn2(self.conv11(x9)))

      return x


if __name__ == "__main__":
    import torch
    inputs = torch.randn((1, 3, 572, 572))
    model = UNet(3,1)
    y = model(inputs)
    # (1, 1, 388, 388) is expected
    print(y.shape)