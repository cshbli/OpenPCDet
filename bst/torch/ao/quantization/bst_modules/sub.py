from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Sub(nn.Module):
    def forward(self, x, y):
        return x - y
