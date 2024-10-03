from dataclasses import dataclass
from typing import Any
from distributions.distributions import WrappedNormal
from models.blocks import *
import torch.distributions as dist
from models.poincare_ball import *

@dataclass
class ConfigEncoder:
    manifold: Any
    data_size: Any
    num_hidden_layers: int
    hidden_dim: int
    prior_iso: bool
    eta: float = 1e-6

@dataclass
class ConfigDecoder:
    manifold: Any
    data_size: Any
    num_hidden_layers: int
    hidden_dim: int 

@dataclass
class ConfigVAE:
    prior_distribution: Any
    likelihood_distribution: Any
    posterior_distribution: Any
    encoder: Any
    decoder: Any
    encoder_config: Any
    decoder_config: Any
    manifold: Any
    c: float = 1.0
    data_size: float = 1
    latent_dim: int = 10
    prior_std: float = 1.0
    learn_prior_std: bool =False

@dataclass
class ConfigTrainer:
    config_vae: Any
    loader_config: Any
    epochs: int = 10
    beta: float = 1.0
    lr: float = 3e-4
    analytical_kl: bool = True
    save_path: str = '/home/ubuntu/HYPERBOLIC/artifacts'

@dataclass
class ConfigLoader:
    batch_size: int = 64
    shuffle: bool = True
    eta: float = 1e-6

def get_args():

    encoder_config = ConfigEncoder(
        manifold=PoincareBall(dim=10),
        data_size=(1, 28, 28),
        num_hidden_layers=2,
        hidden_dim=10,
        prior_iso=False
    )

    decoder_config = ConfigDecoder(
        manifold=PoincareBall(dim=10),
        data_size=(1, 28, 28),
        hidden_dim=10,
        num_hidden_layers=2,
    )

    config_vae = ConfigVAE(
        prior_distribution=WrappedNormal,
        likelihood_distribution=dist.RelaxedBernoulli,
        posterior_distribution=WrappedNormal,
        encoder=EncoderWrapped,
        decoder=DecoderWrapped,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        manifold=PoincareBall(dim=10)
    )

    loader_config = ConfigLoader()

    config_trainer = ConfigTrainer(
        config_vae=config_vae,
        loader_config=loader_config,
    )

    return config_trainer
