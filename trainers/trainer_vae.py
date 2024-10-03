import torch
from tqdm import tqdm
import torch.distributions as dist
from torchvision.utils import save_image
from models.vae import ModelVAE
from .load_data import LoadData
from .utils import plot_loss
import os

class TrainerVAE:
    def __init__(self, config_vae, loader_config, epochs, beta=1.0, lr=1e-3, analytical_kl=False, save_path=None):

        self.config_vae = config_vae
        self.loader_config = loader_config
        self.beta = beta
        self.lr = lr
        self.analytical_kl = analytical_kl
        self.epochs = epochs

        self.loader = LoadData(**loader_config.__dict__)
        self.vae = ModelVAE(**config_vae.__dict__)

        self.optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.lr)
        self.save_path = save_path

        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

    @staticmethod
    def has_analytic_kl(type_p, type_q):
        return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

    def get_reconstruction_loss(self, px_z, x_batch):
        flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
        reconstruction_loss = px_z.log_prob(x_batch.expand(px_z.batch_shape)).view(flat_rest).sum(-1)
        return reconstruction_loss

    def get_kl_loss(self, qz_x, pz, zs):
        if self.has_analytic_kl(type(qz_x), type(pz)) and self.analytical_kl:
            kl_loss = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1)
        else:
            kl_loss = qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)
        return kl_loss

    def train(self):
        train_loader = self.loader.get_train_loader()

        self.vae.train()
        total_loss = 0
        for x_batch, _ in tqdm(train_loader, desc="Training"):
            x_batch.requires_grad = True
            # Step 1: Encoder
            z_params = self.vae.encode(x_batch)  # Get q(z|x) parameters

            # Step 2: Sample from the posterior
            qz_x = self.vae.posterior_distribution(*z_params)
            zs = self.vae.sample(qz_x, K=1)  # Sample from q(z|x)

            # Step 3: Decoder
            rec_params = self.vae.decode(zs)  # Get p(x|z) parameters
            px_z = self.vae.likelihood_distribution(*rec_params)

            # Step 4: Calculate losses
            #reconstruction_loss = self.get_reconstruction_loss(px_z, x_batch)
            pz = self.vae.get_pz()
            kl_loss = self.get_kl_loss(qz_x, pz, zs)

            flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
            reconstruction_loss = px_z.log_prob(x_batch.expand(px_z.batch_shape)).view(flat_rest).sum(-1)

            # ELBO (Negative Evidence Lower Bound)
            loss = -reconstruction_loss.mean(0).sum() #+ self.beta * kl_loss.mean(0).sum()

            # Step 5: Backpropagation and parameter updates
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def test(self):
        test_loader = self.loader.get_test_loader()

        self.vae.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, _ in tqdm(test_loader, desc="Testing"):
                # Step 1: Encoder
                z_params = self.vae.encode(x_batch)  # Get q(z|x) parameters

                # Step 2: Sample from the posterior
                qz_x = self.vae.posterior_distribution(*z_params)
                zs = self.vae.sample(qz_x, K=1)  # Sample from q(z|x)

                # Step 3: Decoder
                rec_params = self.vae.decode(zs)  # Get p(x|z) parameters
                px_z = self.vae.likelihood_distribution(*rec_params)

                # Step 4: Calculate losses
                reconstruction_loss = self.get_reconstruction_loss(px_z, x_batch)
                pz = self.vae.get_pz()
                kl_loss = self.get_kl_loss(qz_x, pz, zs)

                # ELBO (Negative Evidence Lower Bound)
                loss = -reconstruction_loss.mean(0).sum() + self.beta * kl_loss.mean(0).sum()

                total_loss += loss.item()

        return total_loss / len(test_loader)

    def reconstruct(self, epoch):
        """Reconstruct images from the first batch of training data."""
        train_loader = self.loader.get_train_loader()

        # Get the first batch of data
        x_batch, _ = next(iter(train_loader))

        self.vae.eval()
        with torch.no_grad():
            # Step 1: Encoder
            z_params = self.vae.encode(x_batch)  # Get q(z|x) parameters

            # Step 2: Sample from the posterior
            qz_x = self.vae.posterior_distribution(*z_params)
            zs = self.vae.sample(qz_x, K=1)  # Sample from q(z|x)

            # Step 3: Decoder
            rec_params = self.vae.decode(zs)  # Get p(x|z) parameters
            px_z = self.vae.likelihood_distribution(*rec_params)

            reconstructions = rec_params[0].squeeze(0)

        # Select 5 images from the batch
        original_images = x_batch[:5]
        reconstructed_images = reconstructions[:5]

        # Concatenate original and reconstructed images
        comparison = torch.cat([original_images, reconstructed_images])

        # Save the comparison as an image
        save_image(comparison.cpu(), f'{self.save_path}/plot/reconstruction_epoch_{epoch}.png', nrow=5)


    def run(self):
        train_losses, test_losses = [], []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self.train()
            test_loss = self.test()

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Plot and save the loss curves
            plot_loss(train_losses, test_losses, epoch, save_path=self.save_path)

            # Reconstruct and save images
            self.reconstruct(epoch)
