import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.make_block(3, 8, padding=1)
        self.conv2 = self.make_block(8, 16, padding=1)
        self.conv3 = self.make_block(16, 32, padding=1)
        self.conv4 = self.make_block(32, 64, padding=1)
        self.conv5 = self.make_block(64, 64, padding=1)


        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)  # ~ np.reshape
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = CNN(num_classes=10)
    sample_input = torch.rand(2, 3, 224, 224)
    output = model(sample_input)
    print(output)
