import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from models.poincare_ball import PoincareBall
import torch.distributions as dist


class ModelVAE(nn.Module):
    def __init__(
        self,
        prior_distribution,
        likelihood_distribution,
        posterior_distribution,
        encoder,
        decoder,
        encoder_config,
        decoder_config,
        c,
        data_size,
        prior_std,
        latent_dim,
        learn_prior_std,
        manifold,

    ):
        super().__init__()
        
        self.prior_distribution = prior_distribution  # p(z)
        self.likelihood_distribution = likelihood_distribution  # p(x|z)
        self.posterior_distribution = posterior_distribution  # q(z|x)

        self.encoder = encoder(**encoder_config.__dict__)
        self.decoder = decoder(**decoder_config.__dict__)

        self.manifold = manifold

        self.c = c
        self.data_size = data_size
        self.prior_std = prior_std
        self.learn_prior_std = learn_prior_std

        # Prior p(z) mean and log variance
        self._pz_mu = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=learn_prior_std)

        if self.likelihood_distribution == dist.RelaxedBernoulli:
            self.likelihood_distribution.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def get_mean_param(self, params):
        """Helper function to get the mean parameter."""
        if params[0].dim() == 0:
            return params[1]
        else:
            return params[0]

    def get_pz(self):
        """Returns the prior distribution p(z)."""
        pz_mu, log_var, manifold = self.get_pz_params()
        return self.prior_distribution(
            loc=pz_mu,
            scale=log_var,
            manifold=manifold
        )

    def encode(self, x):
        """Encode input x into latent variable z."""
        return self.encoder(x)

    def sample(self, qz_x, K):
        """Sample z from q(z|x) using reparameterization trick."""
        return qz_x.rsample(torch.Size([K]))

    def decode(self, z):
        """Decode latent variable z into reconstruction."""
        return self.decoder(z)

    def forward(self, x, K=1):
        """Forward pass through the VAE: encode -> sample -> decode."""
        # Encode input x to get posterior q(z|x) parameters
        z_params = self.encode(x)

        # Sample from q(z|x)
        qz_x = self.posterior_distribution(*z_params)
        zs = self.sample(qz_x, K)

        # Decode latent samples to get likelihood distribution parameters
        rec_params = self.decode(zs)
        px_z = self.likelihood_distribution(*rec_params)

        return qz_x, px_z, zs

    def reconstruct(self, data):
        """Reconstruct the input data."""
        self.eval()
        with torch.no_grad():
            qz_x = self.posterior_distribution(*self.encode(data))
            px_z_params = self.decode(qz_x.rsample(torch.Size([1])).squeeze(0))
        return self.get_mean_param(px_z_params)

    def generate(self, N, K):
        """Generate new data from the prior distribution p(z)."""
        self.eval()
        with torch.no_grad():
            mean_pz = self.get_mean_param(self.pz_params)
            mean = self.get_mean_param(self.decoder(mean_pz))
            px_z_params = self.decoder(self.posterior_distribution(*self.pz_params).sample(torch.Size([N])))
            means = self.get_mean_param(px_z_params)
            samples = self.likelihood_distribution(*px_z_params).sample(torch.Size([K]))
        return mean, means.view(-1, *means.size()[2:]), samples.view(-1, *samples.size()[3:])

    def get_pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold
