import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import numpy as np
import pyro
import pyro.contrib.gp as gp
from src.utilities_full import BayesOpt
#assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)
np.random.seed(0)

#
# def f(x):
#     return (6 * x - 2)**2 * np.sin(12 * x - 4)
# lower = np.array([0.0])
# upper = np.array([1.])
# #
# solution = BayesOpt().solve(f, [0.4], bounds=(lower,upper), maxfun=8)
#


# def f1(x):
#     return (6 * x[0] - 2)**2 * np.sin(12 * x[1] - 4)
def g1(x):
    return (6 * x[0] - 2) - 1


def f1(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[0] - 4) + 100*max(0, g1(x))**2
def f2(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[0] - 4)



lower = np.array([0.0]*1)
upper = np.array([1.]*1)

# solution1 = BayesOpt().solve(f1, [0], bounds=(lower,upper), print_iteration=True)#, constraints=[g1])

def f1(x):
    return (6 * x[0] - 2)**2 * np.sin(12 * x[1] - 4) + 100*max(0,(6 * x[0] - 2)**2  - 1)**2
def g1(x):
    return -1 * ((6 * x[0] - 2)**2 - 1)
import casadi as ca
def g_known(x):
    return x[0] + ca.sin(x[1]) - 1

def fun_eval(x):
    return f1(x)#, []

lower = np.array([0.0]*2)
upper = np.array([1.]*2)

X = np.random.rand(4, 2)
Y = np.array([[f1(X[i]) for i in range(4)], [g1(X[i]) for i in range(4)]]).T

solution = BayesOpt().solve(objective=fun_eval, xo=lower, bounds=(lower, upper),
                            maxfun=10, N_constraints=None, casadi=True, print_iteration=True,
                            known_constraints=[g_known], idx_integer=[1],
                            select_kernel='Matern52',acquisition='LCB')

print(2)