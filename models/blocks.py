import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from .manifold_layers import *


class EncoderLinear(nn.Module):
    """ Usual encoder """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim, prior_iso, eta=1-5):
        super().__init__()

        self.manifold = manifold
        self.data_size = data_size
        self.eta = eta

        # Initial input layer
        layers = [nn.Sequential(nn.Linear(prod(data_size), hidden_dim), nn.ReLU())]

        # Add hidden layers explicitly
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.enc = nn.Sequential(*layers)

        # Final output layers for mean and variance
        self.fc_mu = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc_logvar = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        # Flatten input data and pass through encoder layers
        e = self.enc(x.view(x.size(0), -1))
        
        # Compute mean and variance (ensure variance is positive with softplus)
        mu = self.fc_mu(e)
        logvar = F.softplus(self.fc_logvar(e)) + self.eta
        
        return mu, logvar, self.manifold


class DecoderLinear(nn.Module):
    """ Usual decoder with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim):
        super().__init__()
        self.data_size = data_size

        # Initial input layer
        layers = [nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.dec = nn.Sequential(*layers)

        # Final output layer to match the data size
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        # Pass latent variable z through the decoder layers
        d = self.dec(z)
        
        # Reshape the output to match the original data size
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)
        
        return mu, torch.ones_like(mu)
    

class EncoderWrapped(nn.Module):
    """ Usual encoder followed by an exponential map with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim, prior_iso, eta=1e-5):
        super().__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.eta = eta

        # Initial input layer
        layers = [nn.Sequential(nn.Linear(prod(data_size), hidden_dim), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.enc = nn.Sequential(*layers)

        # Final output layers for mean and variance
        self.fc_mu = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc_logvar = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        # Flatten input data and pass through encoder layers
        e = self.enc(x.view(x.size(0), -1))
        
        # Compute the mean and apply the exponential map
        mu = self.fc_mu(e)
        mu = self.manifold.expmap0(mu)
        
        # Compute variance (ensure positive variance with softplus)
        logvar = F.softplus(self.fc_logvar(e)) + self.eta
        
        return mu, logvar, self.manifold


class DecoderWrapped(nn.Module):
    """ Usual decoder preceded by a logarithm map with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim):
        super().__init__()
        self.data_size = data_size
        self.manifold = manifold

        # Initial input layer
        layers = [nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.dec = nn.Sequential(*layers)

        # Final output layer to match the data size
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        # Apply logarithm map before passing to the decoder
        z = self.manifold.logmap0(z)

        # Pass through decoder layers
        d = self.dec(z)

        # Reshape the output to match the original data size
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)

        return mu, torch.ones_like(mu)

class DecoderGedesic(nn.Module):
    """ First layer is a Hypergyroplane (Geodesic Layer) followed by usual decoder with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim):
        super().__init__()
        self.data_size = data_size

        # Initial layer is the GeodesicLayer followed by ReLU activation
        layers = [nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.dec = nn.Sequential(*layers)

        # Final output layer to match the data size
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        # Pass latent variable z through the decoder layers
        d = self.dec(z)

        # Reshape the output to match the original data size
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)

        return mu, torch.ones_like(mu)


class EncoderMobius(nn.Module):
    """ Last layer is a Mobius layer with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim, prior_iso, eta=1e-5):
        super().__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.eta = eta

        # Initial input layer followed by ReLU
        layers = [nn.Sequential(nn.Linear(prod(data_size), hidden_dim), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.enc = nn.Sequential(*layers)

        # Mobius Layer for mean computation
        self.fc_mu = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)

        # Final output for variance
        self.fc_logvar = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        # Flatten the input and pass through encoder layers
        e = self.enc(x.view(x.size(0), -1))

        # Compute mean using the Mobius layer
        mu = self.fc_mu(e)

        # Apply the exponential map to the mean
        mu = self.manifold.expmap0(mu)

        # Compute the variance and ensure it's positive using softplus
        logvar = F.softplus(self.fc_logvar(e)) + self.eta
        
        return mu, logvar, self.manifold

class DecoderMobius(nn.Module):
    """ First layer is a Mobius Matrix multiplication with ReLU activation """
    def __init__(self, manifold, data_size, num_hidden_layers, hidden_dim):
        super().__init__()
        self.data_size = data_size

        # Initial Mobius Layer followed by LogZero and ReLU
        layers = [nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), nn.ReLU())]

        # Add hidden layers explicitly with ReLU activation
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.dec = nn.Sequential(*layers)

        # Final output layer to match the data size
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        # Pass the latent variable through decoder layers
        d = self.dec(z)

        # Reshape the output to match the original data size
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)

        return mu, torch.ones_like(mu)

class DecoderBernouilliWrapper(nn.Module):
    """ Wrapper for Bernoulli likelihood """
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z):
        mu, _ = self.dec.forward(z)
        return torch.tensor(1.0).to(z.device), mu