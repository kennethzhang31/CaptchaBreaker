import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from datasets import load_dataset

device = torch.device("mps")

image_width = 200
image_height = 50
latent_dim = 100
channels = 1
epochs = 10

train_dataset = (load_dataset("project-sloth/captcha-images", split='train'))
valid_dataset = (load_dataset("project-sloth/captcha-images", split='validation'))
test_dataset = (load_dataset("project-sloth/captcha-images", split='test'))

transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.Grayscale(num_output_channels=channels),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})
valid_dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})
test_dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

generator = Generator()
discriminator = Discriminator()
adversarial_loss = nn.BCELoss()

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

for epoch in range(epochs):
    for i, batch  in enumerate(train_loader):
        real_image = batch['image'].to(device)
        batch_size = real_image.size(0)

        real_labels = torch.ones(batch_size, 1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
        
        dis_optimizer.zero_grad()
        real_loss = adversarial_loss(discriminator(real_image), real_labels)

        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(z)

        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        dis_optimizer.step()

        gen_optimizer.zero_grad()
        gen_loss = adversarial_loss(discriminator(fake_images), real_labels)
        gen_loss.backward()
        gen_optimizer.step()

        if i % 10 == 0:
            print(f'Epoch [{epoch} / {epochs}], Step: [{i} / {len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {gen_loss.item():.4f}')