import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from config import get_config
from dcgan.models import Discriminator, Generator
from dcgan.operations import init_weights, get_auc

config = get_config()
random.seed(config["seed"])
torch.manual_seed(config["seed"])

device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and config["n_gpus"] > 0) else "cpu"
)

dataset = dset.ImageFolder(
    root=config["dataroot"],
    transform=transforms.Compose(
        [
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
)

real_batch = next(iter(dataloader))

discriminator = Discriminator(config["n_gpus"]).to(device)
generator = Generator(config["n_gpus"]).to(device)

if device.type == "cuda" and config["n_gpus"] > 1:
    discriminator = nn.DataParallel(discriminator, list(range(config["n_gpus"])))
    generator = nn.DataParallel(generator, list(range(config["n_gpus"])))

discriminator.apply(init_weights)
generator.apply(init_weights)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, config["nz"], 1, 1, device=device)
label_real = 1.0
label_fake = 0.0

optimiser_dis = optim.Adam(
    discriminator.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999)
)
optimiser_gen = optim.Adam(
    generator.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999)
)

img_list = []
G_losses = []
D_losses = []
j = 0

print("Starting Training Loop...")
for epoch in range(config["num_epochs"]):
    for i, data in enumerate(dataloader, 0):
        ##################
        # DISCRIMINATOR UPDATE
        ##################

        # reset parameter gradients, send images to gpu, get batch size
        discriminator.zero_grad()
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)

        # create vector of labels, get vector of predictions, calculate and backpropagate error
        label = torch.full(
            size=(b_size,), fill_value=label_real, dtype=torch.float, device=device
        )
        output_D_x = discriminator(real_imgs).view(-1)
        discriminator_error_real = criterion(output_D_x, label)
        discriminator_error_real.backward()

        # D_x refers to D(x) - the mean prediction of the discriminator on real inputs x
        D_x = output_D_x.mean().item()

        # create matrix of random vectors z for generating fake images
        noise = torch.randn(b_size, config["nz"], 1, 1, device=device)
        # generate fake images, create vector of labels, make predictions, backpropagate errors
        fake_imgs = generator(noise)
        label.fill_(label_fake)
        output_D_G_z = discriminator(fake_imgs.detach()).view(-1)
        discriminator_error_fake = criterion(output_D_G_z, label)
        discriminator_error_fake.backward()
        # D_G_z refers to D(G(z)) - the mean prediction of the discriminator on images generated by G on z
        # _1 refers to the mean prediction before an optimisation step
        D_G_z_1 = output_D_G_z.mean().item()

        # sum errors for recording. both errors have backpropagated to gradients already for optimiser step
        discriminator_error = discriminator_error_fake + discriminator_error_real
        optimiser_dis.step()

        ##################
        # GENERATOR UPDATE
        ##################
        generator.zero_grad()
        # can use the above output as it was generated. will predict with discriminator again after step
        # _2 refers to the mean prediction after an optimisation step
        output_D_G_z = discriminator(fake_imgs).view(-1)
        label.fill_(label_real)
        generator_error = criterion(output_D_G_z, label)
        generator_error.backward()
        D_G_z_2 = output_D_G_z.mean().item()
        optimiser_gen.step()

        if i % 50 == 0:
            predictions_real = output_D_x.detach().cpu().numpy()
            predictions_fake = output_D_G_z.detach().cpu().numpy()
            labels_real = np.full(predictions_real.shape, label_real)
            labels_fake = np.full(predictions_fake.shape, label_fake)
            predictions = np.concatenate((predictions_real, predictions_fake))
            labels = np.concatenate((labels_real, labels_fake))
            auc = get_auc(labels, predictions)

            print(
                f"[{epoch+1}/{config['num_epochs']}][{i}/{len(dataloader)}]",
                f"Loss_D: {discriminator_error:.2f}",
                f"Loss_G: {generator_error:.2f}",
                f"D(x): {D_x:.2f}",
                f"D(G(z))_1: {D_G_z_1:.2f} -> D(G(Z))_2: {D_G_z_2:.2f}",
                f"auc: {auc:.2f}",
                sep="\t",
            )

        if i % 500 == 0 or (
            epoch == config["num_epochs"] - 1 and i == (len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        j += 1

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64], padding=5, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
