import torch
import torch.nn as nn

image_height = 50
image_width = 200
latent_dim = 100
channels = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, (4, 4), 1, 0),  # Output: 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1)),  # Output: 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),  # Output: 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, (3, 4), (2, 1), (1, 1)),   # Adjust height: Output 64 x 32 x 33
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels, (3, 5), (2, 1), (1, 1)),  # Adjust width: Output 1 x 50 x 200
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), latent_dim, 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, (3, 5), (2, 1), (1, 1)),  # Output: 64 x 25 x 198
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, (3, 4), (2, 1), (1, 1)),  # Output: 128 x 13 x 197
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, (3, 4), (2, 1), (1, 1)),  # Output: 256 x 7 x 196
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, (3, 4), (2, 1), (1, 1)),  # Output: 512 x 4 x 195
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Use a smaller kernel size to avoid the error
            nn.Conv2d(512, 1, (4, 4), 1, 0),  # Final output: 1 x 1 x 192
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.model(img)
        # Reduce the last dimension by averaging
        img = img.mean(dim=[2, 3])  # Output shape will be [64, 1]
        return img

