import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from .poincare_ball import PoincareBall
from geoopt import ManifoldParameter
from.euclidean_manifold import Euclidean


# Esta classe é a camada base para definir uma camada em uma variedade Riemanniana
class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm

        if self.over_params:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))

        self.reset_parameters()

    # Calcula os pesos projetados na variedade. Os pesos são transportados para o espaço tangente em torno do bias usando transporte paralelo.
    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight) # weight \in T_0 => weight \in T_bias

    # Se over_param for verdadeiro, o bias é retornado diretamente. Caso contrário, o bias é reparametrizado no espaço da variedade usando o mapa exponencial.
    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias) # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))

# Esta camada estende a RiemannianLayer e calcula a distância geodésica de uma entrada até um hiperplano definido pelos pesos.
class GeodesicLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight,
                                               signed=True, norm=self.weight_norm)
        return res

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(Linear, self).__init__(
            in_features,
            out_features,
        )

#  Multiplicação de Möbius entre os pesos e as entradas.
class MobiusLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res

# Mapa exponencial
class ExpZero(nn.Module):
    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)

# Mapa logaritmo
class LogZero(nn.Module):
    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)
    


    


