import torch
from torch import nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Args:
        input: An integer indicating number of input channels.
        hidden: An integer indicating number of hidden units between layers.
        output: An integer indicating number of output units.
    """
    def __init__(self, input: int, hidden: int, output: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input,
                out_channels = hidden,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden,
                out_channels = hidden,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                ),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels = hidden,
                out_channels = hidden,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden,
                out_channels = hidden,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden*16*16, 
                out_features=output,
                ),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block2(self.conv_block1(x)))
