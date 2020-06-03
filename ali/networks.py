import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=(4, 4), stride=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.02, inplace=True),

            # 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.02, inplace=True),

            # 13 x 13
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=(5, 5), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.02, inplace=True),

            # 29 x 29
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.02, inplace=True),

            # 63 x 63
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=(2, 2), stride=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.02, inplace=True),

            # 64 x 64
            nn.Conv2d(in_channels=ngf, out_channels=nc, kernel_size=(1, 1), stride=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class Encoder(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ngf, kernel_size=(2, 2), stride=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.02, inplace=True),

            # 63 x 63
            nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.02, inplace=True),

            # 29 x 29
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=(5, 5), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.02, inplace=True),

            # 13 x 13
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.02, inplace=True),

            # 4 x 4
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=(4, 4), stride=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.02, inplace=True),

            # 1 x 1
            nn.Conv2d(in_channels=ngf * 8, out_channels=2 * nz, kernel_size=(1, 1), stride=1, bias=False)
        )

    def forward(self, x):
        out = self.main(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, nz, dropout=0.2):
        super(Discriminator, self).__init__()

        self.x2feat = nn.Sequential(
            # 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(2, 2), stride=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            # 63 x 63
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            # 29 x 29
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(5, 5), stride=2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            # 13 x 13
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 4, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            # 4 x 4
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=(4, 4), stride=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            # 1 x 1
        )

        self.z2feat = nn.Sequential(
            nn.Conv2d(in_channels=nz, out_channels=ndf * 8, kernel_size=(1, 1), stride=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=ndf * 8, out_channels=ndf * 8, kernel_size=(1, 1), stride=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout)
        )

        self.feat2out = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 16, out_channels=ndf * 32, kernel_size=(1, 1), stride=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=ndf * 32, out_channels=ndf * 32, kernel_size=(1, 1), stride=1, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=ndf * 32, out_channels=1, kernel_size=(1, 1), stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        feat_x = self.x2feat(x)
        feat_z = self.z2feat(z)
        feat = torch.cat([feat_x, feat_z], dim=1)
        out = self.feat2out(feat)
        return out
