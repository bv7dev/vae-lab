import math, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_size: list[int] = None, latent_dim: int = None, conv_channels: list[int] = None, ffn_layers: list[int] = None):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.ffn_layers = ffn_layers
        self.loss_history = { "loss_total": [], "mse_loss": [], "kld_loss": [],
                              "mse_weight": [], "kld_weight": [], "lr": [] }
        if not None in [input_size, latent_dim, conv_channels]:
            self.current_epoch = 0
            self._init_modules()

    def load(self, model_name: str, model_dir: str):
        model_path = os.path.join(model_dir, model_name)
        path = os.path.join(model_path, model_name)
        with open(path + ".json", "r") as fp:
            metadata = json.load(fp)
        self.input_size   = metadata["input_size"]
        self.latent_dim    = metadata["latent_dim"]
        self.conv_channels = metadata["conv_channels"]
        self.ffn_layers    = metadata["ffn_layers"]
        self.current_epoch = metadata["current_epoch"]
        self.loss_history  = metadata["loss_history"]
        self._init_modules()
        self.load_state_dict(torch.load(path + ".pt", weights_only=True))
        self.save(model_name, model_dir, self.current_epoch, cp=True)

    def save(self, model_name: str, model_dir: str, epoch: int, cp=False):
        model_path = os.path.join(model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        path = os.path.join(model_path, model_name + ("_checkpoint_" + str(epoch) if cp else ""))
        metadata = {
            "input_size"   : self.input_size,
            "latent_dim"   : self.latent_dim,
            "conv_channels": self.conv_channels,
            "ffn_layers"   : self.ffn_layers,
            "current_epoch": epoch,
            "loss_history" : self.loss_history,
        }
        with open(path + ".json", "w") as fp:
            json.dump(metadata, fp)
        with torch.no_grad():
            torch.save(self.state_dict(), path + ".pt")
    
    def loss(self, orig, recon, z_mean, z_log_var, mse_weight, kld_weight, lr):
        mse_loss = F.mse_loss(recon, orig)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1))
        loss_total: torch.Tensor = mse_loss * mse_weight + kld_loss * kld_weight
        self._record_loss(loss_total.detach().item(), mse_loss.detach().item(), kld_loss.detach().item(), mse_weight, kld_weight, lr)
        return loss_total

    def encode(self, x):
        x = self.ffn_enc(torch.flatten(self.conv(x), start_dim=1))
        return self.ffn_mean(x), self.ffn_log_var(x)

    def decode(self, z):
        z = self.ffn_dec(z).view(-1, *self.conv_size[1:])
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

    def _record_loss(self, loss_total, mse_loss, kld_loss, mse_weight, kld_weight, lr):
        self.loss_history["loss_total"].append(loss_total)
        self.loss_history["mse_loss"].append(mse_loss)
        self.loss_history["kld_loss"].append(kld_loss)
        self.loss_history["mse_weight"].append(mse_weight)
        self.loss_history["kld_weight"].append(kld_weight)
        self.loss_history["lr"].append(lr)


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _init_modules(self):
        conv_channels = self.conv_channels.copy()
        ffn_layers = self.ffn_layers.copy()

        # encoder
        channels = self.input_size[1]
        self.conv = nn.Sequential()

        for cc in conv_channels:
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(channels, cc, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(cc),
                    nn.GELU() ) )
            channels = cc
        
        self.conv_size = self.conv(torch.rand(self.input_size)).size()

        # feed-forward network
        flat_size = math.prod(self.conv_size[1:])

        layer_size = flat_size
        self.ffn_enc = nn.Sequential()
        for ls in ffn_layers:
            self.ffn_enc.append(
                nn.Sequential(
                    nn.Linear(layer_size, ls),
                    nn.BatchNorm1d(ls),
                    nn.GELU() ) )
            layer_size = ls

        # latent sampling layers
        self.ffn_mean = nn.Linear(layer_size, self.latent_dim)
        self.ffn_log_var = nn.Linear(layer_size, self.latent_dim)

        # decoder ffn
        ffn_layers.reverse()
        ffn_layers.append(flat_size)
        layer_size = self.latent_dim
        self.ffn_dec = nn.Sequential()
        for ls in ffn_layers:
            self.ffn_dec.append(
                nn.Sequential(
                    nn.Linear(layer_size, ls),
                    nn.BatchNorm1d(ls),
                    nn.GELU() ) )
            layer_size = ls

        # decoder conv
        channels = conv_channels.pop()
        conv_channels.reverse()
        conv_channels.append(self.input_size[1])
        self.deconv = nn.Sequential()

        for i, cc in enumerate(conv_channels):
            self.deconv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, cc, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(cc),
                    nn.GELU() if i < (len(conv_channels) - 1) else nn.Sequential(
                        nn.ConvTranspose2d(cc, cc, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.Conv2d(cc, cc, kernel_size=3, stride=2, padding=1),
                        nn.Sigmoid()
                    ) ) )
            channels = cc
        
        self.deconv_size = self.deconv(torch.rand(self.conv_size)).size()

        assert(self.deconv_size == torch.Size(self.input_size))
        
        self.conv.apply(self._init_weights)
        self.ffn_enc.apply(self._init_weights)
        self.ffn_mean.apply(self._init_weights)
        self.ffn_log_var.apply(self._init_weights)
        self.ffn_dec.apply(self._init_weights)
        self.deconv.apply(self._init_weights)


# Test VAE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import *

    images = torch.rand(INPUT_SIZE)
    vae = VAE(INPUT_SIZE, LATENT_SIZE, CONV_CHANNELS, FFN_LAYERS)

    recon, z, z_mean, z_log_var = vae(images)

    print("loss:", vae.loss(images, recon, z_mean, z_log_var, 1, 0.1, 1e-4))

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(images[0].view(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH).permute(1, 2, 0).detach().numpy())
    axes[1].imshow(recon[0].view(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH).permute(1, 2, 0).detach().numpy())

    plt.show()

    # TODO: Save and load model test