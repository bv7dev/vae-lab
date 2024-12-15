import math, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_shape: list[int] = None, latent_dim: int = None, conv_channels: list[int] = None):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.loss_history = { "loss_total": [], "mse_loss": [], "kld_loss": [] }
        if not None in [input_shape, latent_dim, conv_channels]:
            self.current_epoch = 0
            self._init_modules()

    def load(self, model_name: str, model_dir: str):
        path = os.path.join(model_dir, model_name)
        with open(path + ".json", "r") as fp:
            metadata = json.load(fp)
        self.input_shape   = metadata["input_size"]
        self.latent_dim    = metadata["latent_dim"]
        self.conv_channels = metadata["conv_channels"]
        self.current_epoch = metadata["current_epoch"]
        self.loss_history  = metadata["loss_history"]
        self._init_modules()
        self.load_state_dict(torch.load(path + ".pt", weights_only=True))

    def save(self, model_name: str, model_dir: str, epoch: int):
        path = os.path.join(model_dir, model_name)
        with open(path + ".json", "w") as fp:
            json.dump({
                "input_size"   : self.input_shape,
                "latent_dim"   : self.latent_dim,
                "conv_channels": self.conv_channels,
                "current_epoch": epoch,
                "loss_history" : self.loss_history,
                }, fp)
        torch.save(self.state_dict(), path + ".pt")
    
    def loss(self, orig, recon, z_mean, z_log_var, mse_weight=1, kld_weight=1):
        mse_loss = F.mse_loss(recon, orig)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1))
        loss_total = mse_weight*mse_loss + kld_weight*kld_loss
        self._record_loss(loss_total.item(), mse_loss.item(), kld_loss.item())
        return loss_total

    def encode(self, x):
        x = torch.flatten(self.conv(x), start_dim=1)
        return self.fc_mean(x), self.fc_log_var(x)

    def decode(self, z):
        z = self.fc_deconv(z).view(-1, *self.conv_size[1:])
        return self.deconv(z)

    def forward(self, image_batch):
       z_mean, z_log_var = self.encode(image_batch)
       z = self.reparameterize(z_mean, z_log_var)
       return self.decode(z), z, z_mean, z_log_var
    
    @staticmethod
    def reparameterize(z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return std * eps + z_mean

    def _record_loss(self, loss_total, mse_loss, kld_loss):
        self.loss_history["loss_total"].append(loss_total)
        self.loss_history["mse_loss"].append(mse_loss)
        self.loss_history["kld_loss"].append(kld_loss)

    def _init_modules(self):
            conv_channels = self.conv_channels.copy()

            # encoder
            channels = self.input_shape[1]
            self.conv = nn.Sequential()

            for cc in conv_channels:
                self.conv.append(
                    nn.Sequential(
                        nn.Conv2d(channels, cc, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(cc),
                        nn.LeakyReLU() ) )

                channels = cc
            
            self.conv_size = self.conv(torch.rand(self.input_shape)).size()

            # latent
            flat_size = math.prod(self.conv_size[1:])
            self.fc_mean = nn.Linear(flat_size, self.latent_dim)
            self.fc_log_var = nn.Linear(flat_size, self.latent_dim)

            # decoder
            channels = conv_channels.pop()
            conv_channels.reverse()
            conv_channels.append(self.input_shape[1])

            self.fc_deconv = nn.Linear(self.latent_dim, flat_size)
            self.deconv = nn.Sequential()

            for i, cc in enumerate(conv_channels):
                self.deconv.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(channels, cc, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(cc),
                        nn.LeakyReLU() if i < (len(conv_channels) - 1) else nn.Sigmoid() ) )
                
                channels = cc
    
# Test VAE
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images = torch.rand(8, 3, 32, 32)
    vae = VAE(images.size(), 512, [6, 12])

    recon, z, z_mean, z_log_var = vae(images)

    print("loss:", vae.loss(images, recon, z_mean, z_log_var))

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(images[0].view(3, 32, 32).permute(1, 2, 0).detach().numpy())
    axes[1].imshow(recon[0].view(3, 32, 32).permute(1, 2, 0).detach().numpy())

    plt.show()

