import pyro
import torch
import pytest
import sobol_seq
import numpy as np
import pymoo.optimize
import pyro.contrib.gp as gp
from SI_BOpt.src.utilities_full import BayesOpt
from pymoo.core.mixed import MixedVariableGA
from SI_BOpt.src.extra_utils import MixedRBF, MixedMatern32, MixedMatern52, ConstrainedProblem
from SI_BOpt.optimization_toy_functions import Constrained_MIP, Rosenbrock_disk_constrained, Matyas

problem = Rosenbrock_disk_constrained()
n_initial = 8
idx_integer = 1
x_dim = problem.x_dim
lower_b = problem.lower_bound
upper_b = problem.upper_bound


@pytest.fixture
def x_int_sobol():
    bounds = (lower_b, upper_b)
    x = torch.from_numpy(sobol_seq.i4_sobol_generate(x_dim, n_initial) *
                         (bounds[1] - bounds[0]) + bounds[0])
    x[..., idx_integer] = torch.round(x[..., idx_integer])
    return x


def test_sobol_integer(x_int_sobol):
    x = x_int_sobol
    x_shape = list(x.shape)
    expect_shape = [n_initial, 2]
    check_shape = (x_shape == expect_shape)
    check_lower_b = (x >= torch.tensor(lower_b)).all().item()
    check_upper_b = (x <= torch.tensor(upper_b)).all().item()
    x_int = x[..., idx_integer].detach().tolist()
    check_int = all([(np.round(i) - (i)) == 0.0 for i in x_int])
    results = all([check_shape, check_lower_b, check_upper_b, check_int])

    assert results


@pytest.mark.parametrize('kernel', [MixedRBF, MixedMatern32, MixedMatern52])
def test_kernel_int(kernel, x_int_sobol):
    x = x_int_sobol
    x_np = x.detach().numpy()
    f_obj = problem.f_obj
    y = torch.tensor([f_obj(x_np[i]) for i in range(x.shape[0])])
    gp_kernel = kernel(input_dim=x_dim, lengthscale=torch.ones(x_dim), integer_dims=idx_integer)
    gp_model = gp.models.GPRegression(x, y, gp_kernel, noise=torch.tensor(0.1), jitter=1.0e-4)
    gp_model.set_data(x, y)
    # optimize the GP hyperparameters using Adam with lr=0.001
    pyro.get_param_store().clear()
    optimizers = torch.optim.Adam(gp_model.parameters(), lr=0.001)
    _ = gp.util.train(gp_model, optimizers)

    predict_1 = gp_model(torch.tensor(np.array([[0., 1.]])))[0].item()
    predict_2 = gp_model(torch.tensor(np.array([[0., 0.9]])))[0].item()
    predict_3 = gp_model(torch.tensor(np.array([[0., 1.3]])))[0].item()
    check_12 = (predict_1 == predict_2)
    check_13 = (predict_1 == predict_3)

    assert (check_12 == check_13)


@pytest.mark.parametrize("algorithm, termination",
                         [
                             (MixedVariableGA(pop=300), ('n_evals', 3000))
                         ])
def test_constrained_opt_pymoo(algorithm, termination):
    res = pymoo.optimize.minimize(ConstrainedProblem(x_dim=problem.x_dim,
                                                     x_bounds=(problem.lower_bound, problem.upper_bound),
                                                     f_objs=[problem.f_obj],
                                                     constraints=problem.constraints,
                                                     ),
                                  algorithm,
                                  termination,
                                  seed=9,
                                  verbose=False
                                  )
    print(f"\nBest solution found: x={res.X}, f(x)={res.F}, [g(x)]={res.G}")
    print(f"Actual optimal point: x={problem.x_opt}, f(x)={problem.f_opt}, "
          f"[g(x)]={[problem.g1(problem.x_opt)]}")

    x_found = np.array(list(res.X.values()))
    x_true = problem.x_opt

    check = (np.linalg.norm(x_found - x_true) <= 2e-2)
    assert check


@pytest.mark.parametrize('prob',
                         [Matyas(), Constrained_MIP()])   # [Unconstrained, constrained]
def test_BO_integer_known_constr(prob):
    sol = BayesOpt().solve(objective=prob.f_obj, bounds=(prob.lower_bound, prob.upper_bound+1),
                           maxfun=15, N_initial=5, select_kernel='Matern52', acquisition='EI', casadi=True,
                           N_constraints=0, known_constraints=prob.constraints, idx_integer=[1])
    x_opt = sol.x
    x_expected = prob.x_opt
    check = (np.linalg.norm(x_opt - x_expected) <= 1e-2)
    assert check


@pytest.mark.parametrize('prob', [Constrained_MIP()])
def test_BO_integer_unknown_constr(prob):
    f_eval = lambda x: (prob.f_obj(x), [g(x) for g in prob.constraints])
    sol = BayesOpt().solve(objective=f_eval, bounds=(prob.lower_bound, prob.upper_bound),
                           maxfun=15, N_initial=5, select_kernel='Matern52', acquisition='EI', casadi=True,
                           N_constraints=prob.n_constr, known_constraints=None, idx_integer=[1])
    x_opt = sol.x
    x_expected = prob.x_opt
    check = (np.linalg.norm(x_opt - x_expected) <= 1e-2)
    assert check
