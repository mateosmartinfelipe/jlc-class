import torch
import torch.nn as nn

image = torch.rand(3, 10, 20)
d0 = image.nelement()

"""01L – Gradient descent and the backpropagation algorithm"""


class MyNet(nn.Module):
    def __init__(self, d0, d1, d2, d3) -> None:
        super().__init__()
        self.m0 = nn.Linear(d0, d1)
        self.m1 = nn.Linear(d1, d2)
        self.m2 = nn.Linear(d2, d3)

    def forward(self, x):
        z0 = x.view(-1)  ## flatten an input tensor
        s1 = self.m0(z0)
        z1 = torch.relu(s1)
        s2 = self.m1(z1)
        z2 = torch.relu(s2)
        s3 = self.m2(z2)
        return s3

    """https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md"""

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


model = MyNet(d0, 60, 40, 10)
print(model(image))
