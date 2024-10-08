import torch
from geoopt.manifolds import PoincareBall as PoincareBallParent
from geoopt.manifolds.stereographic.math import _lambda_x, arsinh, tanh

MIN_NORM = 1e-15

class PoincareBall(PoincareBallParent):

    def __init__(self, dim, c=1.0):
        super().__init__(c)
        self.register_buffer("dim", torch.as_tensor(dim, dtype=torch.int))

    def proju0(self, u):
        return self.proju(self.zero.expand_as(u), u)

    @property
    def coord_dim(self):
        return int(self.dim)

    @property
    def device(self):
        return self.c.device

    @property
    def zero(self):
        return torch.zeros(1, self.dim).to(self.device)
    
    # Calcula o determinante logartimico do mapeamento exponenciol
    def logdetexp(self, x, y, is_vetor=False, keepdim=False):
        d = self.norm(x, y, keepdim=keepdim) if is_vetor else self.dist(x, y, keepdim=keepdim)
        return (self.dim - 1) * (torch.sinh(self.c.sqrt()*d) / self.c.sqrt() / d).log()

    # Calcula o produto interno entre dois vetores u e v no espaço tangente da bola de Poincaré
    def inner(self, x, u, v=None, *, keepdim=False, dim=-1):
        if v is None: v = u
        return _lambda_x(x, self.c, keepdim=keepdim, dim=dim) ** 2 * (u *v).sum(dim=dim, keepdim=keepdim)
    
    # Implementa o mapa exponencial em coordenadas polares, que transforma um vetor tangente u e uma distância r para um ponto na bola de Poincaré a partir de x.
    def expmap_polar(self, x, u, r, dim=-1):
        sqrt_c = self.c**0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = (
            tanh(sqrt_c/2**r)
            * u
            / (sqrt_c*u_norm)
        )
        gamma_1 = self.mobius_add(x, second_term, dim=dim)
        return gamma_1

    # Este método calcula a distância normalizada de um ponto x a um hiperplano, definido por um vetor normal a e um ponto p no espaço hiperbólico.
    def normdist2plane(self, x, a, p, keepdim=False, signed=False, dim=1, norm=False):
        c = self.c
        sqrt_c = c ** 0.5
        diff = self.mobius_add(-p, x, dim=dim)
        diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        sc_diff_a = (diff* a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            sc_diff_a = sc_diff_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * sc_diff_a
        denom = (1-c*diff_norm2)*a_norm
        res = arsinh(num/denom.clamp_min(MIN_NORM)) // sqrt_c
        if norm:
            res = res * a_norm
        return res
    

class PoincareBallExact(PoincareBall):
    __doc__ = r"""
    The implementation of retraction is an exact exponential map, this retraction will be used in optimization
    """

    retr_transp = PoincareBall.expmap_transp
    transp_follow_retr = PoincareBall.transp_follow_expmap
    retr = PoincareBall.expmap

    def extra_repr(self):
        return "exact"


