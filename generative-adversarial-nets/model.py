import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange, reduce, repeat

class Generator(nn.Module):

    def __init__(self, input_dim=100, hidden_dim=1200, output_dim=28 * 28):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(), )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self, input_dim=28 * 28, hidden_dim=240, output_dim=1):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(), )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':

    # define generator, discriminator, and optimizers
    generator = Generator()
    optGen = torch.optim.SGD(params=generator.parameters(),lr=0.1)

    discriminator = Discriminator()
    optDisc = torch.optim.SGD(params=discriminator.parameters(),lr=0.1)

    lossFunc = nn.BCELoss()
    