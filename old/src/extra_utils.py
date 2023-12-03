import numpy as np
import torch
import functools
import warnings
from pyro.contrib.gp.kernels.isotropic import RBF, Matern32, Matern52
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


class PenaltyFunctions:
    def __init__(self, f, g, type_penalty='l2', mu=100):
        self.f = f
        self.g = g
        self.type_p = type_penalty
        self.aug_obj = self.augmented_objective(mu)

    def create_quadratic_penalized_objective(self, mu, order, x):

        obj = self.f(x)
        n_con = len(self.g)
        for i in range(n_con):
            obj += mu * max(self.g[i](x), 0) ** order

        return obj

    def augmented_objective(self, mu):
        """

        :param mu: The penalized parameter
        :type mu: float
        :return:  obj_aug
        :rtype:   function
        """
        if self.type_p == 'l2':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L2 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 2)
        elif self.type_p == 'l1':
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'L1 penalty is used with parameter ' + str(mu))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu, 1)
        else:
            mu_new = 100
            warnings.formatwarning = custom_formatwarning

            warnings.warn(
                'WARNING: Penalty type is not supported. L2 penalty is used instead with parameter ' + str(mu_new))

            obj_aug = functools.partial(self.create_quadratic_penalized_objective,
                                        mu_new, 2)
        return obj_aug


class MixedRBF(RBF):
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, integer_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)
        self.integer_dims = integer_dims

    def forward(self, X, Z=None, diag=False):
        X[..., self.integer_dims] = torch.round(X[..., self.integer_dims])
        if Z is None:
            Z = X
        Z[..., self.integer_dims] = torch.round(Z[..., self.integer_dims])
        output = super().forward(X, Z, diag)
        return output


class MixedMatern32(Matern32):
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, integer_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)
        self.integer_dims = integer_dims

    def forward(self, X, Z=None, diag=False):
        X[..., self.integer_dims] = torch.round(X[..., self.integer_dims])
        if Z is None:
            Z = X
        Z[..., self.integer_dims] = torch.round(Z[..., self.integer_dims])
        output = super().forward(X, Z, diag)
        return output


class MixedMatern52(Matern52):
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, integer_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)
        self.integer_dims = integer_dims

    def forward(self, X, Z=None, diag=False):
        X[..., self.integer_dims] = torch.round(X[..., self.integer_dims])
        if Z is None:
            Z = X
        Z[..., self.integer_dims] = torch.round(Z[..., self.integer_dims])
        output = super().forward(X, Z, diag)
        return output


class ConstrainedProblem(ElementwiseProblem):
    def __init__(self, x_dim, f_objs, x_bounds, integer_idx=None, constraints=None, **kwargs):

        if constraints is None:
            constraints = []

        if integer_idx is None:
            integer_idx = []

        x_types = {}
        for i in range(x_dim):
            if i in integer_idx:
                x_types[i] = Integer(bounds=(x_bounds[0][i], x_bounds[1][i]))
            else:
                x_types[i] = Real(bounds=(x_bounds[0][i], x_bounds[1][i]))

        super().__init__(vars=x_types, n_obj=len(f_objs), n_ieq_constr=len(constraints), n_eq_constr=0, **kwargs)
        self.f = f_objs
        self.g = constraints

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array([x[i] for i in range(self.n_var)])
        out["F"] = [f(x) for f in self.f]
        if self.g:
            out["G"] = [g(x) for g in self.g]
