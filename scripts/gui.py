import torch

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import model, util

class VAE_GUI:
    def __init__(self, vae, latent_size, device, num_sliders=4):
        self.vae = vae
        self.latent_size = latent_size
        self.device = device
        self.num_sliders = num_sliders
        self.latent_vars = torch.randn((1, latent_size))

        self.root = tk.Tk()
        self.root.title("VAE Latent Space Explorer")

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.sliders = []
        for i in range(num_sliders):
            slider = tk.Scale(self.control_frame, from_=-3, to=3, resolution=0.01, orient=tk.HORIZONTAL, label=None, length=600, )
            slider.set(self.latent_vars[0, i].item())
            slider.pack()
            slider.bind("<ButtonRelease-1>", lambda e: self.generate_image())
            self.sliders.append(slider)

        self.randomize_button = tk.Button(self.control_frame, text="Randomize Latents", command=self.randomize_sliders)
        self.randomize_button.pack()

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.generate_image()
    
    def randomize_sliders(self):
        self.latent_vars = torch.randn((1, self.latent_size))
        for i, slider in enumerate(self.sliders):
            slider.set(self.latent_vars[0, i].item())
        self.generate_image()

    def generate_image(self, scale=12):
        z = self.latent_vars.to(self.device)

        for i, slider in enumerate(self.sliders):
            z[0, i] = slider.get()
        with torch.no_grad():
            recon = self.vae.decode(z).cpu().numpy()
        
        if len(recon[0]) == 1:
            recon_image = (recon[0].squeeze() * 255).astype(np.uint8)
            image = Image.fromarray(recon_image, mode='L').resize((recon_image.shape[1] * scale, recon_image.shape[0] * scale), Image.NEAREST)
        elif len(recon[0]) == 3:
            recon_image = (recon[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(recon_image, mode='RGB').resize((recon_image.shape[1] * scale, recon_image.shape[0] * scale), Image.NEAREST)
        else:
            raise ValueError("Invalid number of channels in image")
        
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    vae = model.VAE()
    vae.load("vae-celeba-64x64", "./models/")
    vae.eval()

    device = util.get_device()
    vae.to(device)

    VAE_GUI(vae, vae.latent_dim, device, num_sliders=12).run()