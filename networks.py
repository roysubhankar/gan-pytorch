import torch.nn as nn

class DCGenerator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(DCGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=(4, 4), stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=(4, 4), stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=(4, 4), stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=(4, 4), stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc,
                               kernel_size=(4, 4), stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

# define the discriminator

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(4, 4),
                      stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(4, 4),
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(4, 4),
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=(4, 4),
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=(4, 4),
                      stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out