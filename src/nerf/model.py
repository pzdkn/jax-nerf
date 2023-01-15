import typing as t
from flax import linen as nn

class MLP(nn.Module):
    features: t.Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for features in self.features:
            x = nn.Dense(features)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.relu(x)
        return x

class MiniResNet(nn.Module):
    features: t.Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for features in self.features:
            x_ = nn.Dense(features)(x)
            x = nn.relu(x_) + x 
        x = nn.Dense(self.output_dim)(x)
        return x 

class CNN(nn.Module):
    features: t.Sequence[int]
    kernel_sizes: t.Sequence[t.Sequence[int]]
    output_dim: int 

    @nn.compact
    def __call__(self, x):
        for features, kernel_size in zip(self.features, self.kernel_sizes):
            x = nn.Conv(features=features, kernel_size=kernel_size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
