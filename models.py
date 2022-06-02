import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 1E-5     # Same as PyTorch

        # Trainable parameters
        self.beta = torch.nn.Parameter(data=torch.tensor(0.0))
        self.gamma = torch.nn.Parameter(data=torch.tensor(1.0))
        self.register_parameter('beta', self.beta)
        self.register_parameter('gamma', self.gamma)

    def forward(self, x):
        batch_mean = torch.mean(x)
        batch_var = torch.var(x, unbiased=False)
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.epsilon)
        return self.gamma * x_hat + self.beta


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super().__init__()

        k = (3, 3)
        s = (2 if in_size * 2 == out_size else 1)

        layers = [nn.Conv2d(in_size, out_size, kernel_size=k, stride=s, padding=1)]
        if batch_norm:
            layers.append(BatchNorm())
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(out_size, out_size, kernel_size=k, padding=1))
        if batch_norm:
            layers.append(BatchNorm())
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MnistModel(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()

        self.batch_norm = batch_norm

        layers = [
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
        ]
        for _ in range(2):
            if batch_norm:
                layers.append(BatchNorm())
            layers += [
                nn.Sigmoid(),
                nn.Linear(100, 100)
            ]
        if batch_norm:
            layers.append(BatchNorm())
        layers += [
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Cifar10Model(nn.Module):
    def __init__(self, n, batch_norm=False):
        super().__init__()

        self.batch_norm = batch_norm

        num_layers = {20, 32, 44, 56, 110}
        assert n in num_layers, f'N must be in {list(sorted(num_layers))}'
        k = (n - 2) // 6

        # 3x32x32 -> 16x32x32
        layers = [DoubleConvBlock(3, 16, batch_norm=batch_norm)]
        layers += [DoubleConvBlock(16, 16, batch_norm=batch_norm) for _ in range(k)]

        # 16x32x32 -> 32x16x16
        layers += [DoubleConvBlock(16, 32, batch_norm=batch_norm)]
        layers += [DoubleConvBlock(32, 32, batch_norm=batch_norm) for _ in range(k - 1)]

        # 32x16x16 -> 64x8x8
        layers += [DoubleConvBlock(32, 64, batch_norm=batch_norm)]
        layers += [DoubleConvBlock(64, 64, batch_norm=batch_norm) for _ in range(k - 1)]

        layers += [
            nn.AvgPool2d(8),        # 64x8x8 -> 64x1x1
            nn.Flatten(),
            nn.Linear(64, 10)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
