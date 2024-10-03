import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all


class WrappedNormal(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        self.manifold.assert_check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        """
        Amostra no espaço tangente
        Move o vetor com transporte paralelo
        Mapeia pra variedade novamente
        """
        shape = self._extended_shape(sample_shape)
        
        # is a random vector sampled from a standard normal distribution in the Euclidean tangent space at the origin of the manifold.
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.device)
        self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)

        # hyperbolic scale
        v = v / self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        
        # This step moves the vector v from the tangent space at the origin to the tangent space at loc
        u = self.manifold.transp(self.manifold.zero, self.loc, v)

        # projects the vector u from the tangent space at loc onto the manifold itself.
        z = self.manifold.expmap(self.loc, u)

        return z

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)
        
    def log_prob(self, x):
        shape = x.shape
        loc = self.loc.unsqueeze(0).expand(x.shape[0], *self.batch_shape, self.manifold.coord_dim)
        if len(shape) < len(loc.shape): x = x.unsqueeze(1)

        # mapeia o ponto x para o espaço tangente
        v = self.manifold.logmap(loc, x)

        # move v para o centro do manifold com transporte paralelo
        v = self.manifold.transp(loc, self.manifold.zero, v)

        # reecala v para considerar a curvatura da variedade
        u = v * self.manifold.lambda_x(self.manifold.zero, keepdim=True)

        # normal distribution is used to compute the log-probability of the vector u in Euclidean space
        norm_pdf = Normal(torch.zeros_like(self.scale), self.scale).log_prob(u).sum(-1, keepdim=True)

        # the determinant of the Jacobian of the exponential map measures how much the manifold distorts areas of space as you move from the tangent space to the manifold.
        logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)

        # correção de distorcao do mapeamente euclidiano hiperbolico
        result = norm_pdf - logdetexp
        return result






