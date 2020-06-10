import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid
import math


class MS_task(nn.Module):
    def __init__(self, input_ch=512):
        super(MS_task, self).__init__()
        self.conv = nn.Conv2d(input_ch, 100, kernel_size=1, padding=0)
        self.fc = (nn.Linear(7 * 7 * 100, 49))

        # m = self.conv
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n)) # xavier initialization instead of kaiming_normal

    def forward(self, x):
        #x = sigmoid(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
